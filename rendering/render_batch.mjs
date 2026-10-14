#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import http from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// Optional override for a Playwright install outside node_modules (e.g. an npx cache on a Mac).
const macPlaywright = process.env.RENDER_PLAYWRIGHT_MJS || path.join(os.homedir(), '.npm', '_npx', 'playwright', 'index.mjs');
const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

export function parseArgs(argv) {
  return Object.fromEntries(argv.map(value => {
    const split = value.indexOf('=');
    if (split < 1) throw new Error(`Expected name=value argument: ${value}`);
    return [value.slice(0, split), value.slice(split + 1)];
  }));
}

export function launchOptions(backend, executablePath) {
  let flags;
  if (backend === 'swiftshader') {
    flags = ['--disable-gpu', '--use-gl=angle', '--use-angle=swiftshader', '--allow-file-access-from-files'];
  } else if (backend === 'metal') {
    flags = ['--use-angle=metal', '--ignore-gpu-blocklist', '--allow-file-access-from-files'];
  } else {
    throw new Error(`Unsupported backend: ${backend}`);
  }
  return { headless: true, ...(executablePath ? { executablePath } : {}), args: flags };
}

export function checkBackend(backend, info) {
  if (!info.ok) throw new Error('Renderer has no WebGL context');
  if (backend === 'swiftshader' &&
      (!/swiftshader/i.test(info.unmaskedRenderer || '') || /nvidia|amd|intel/i.test(info.unmaskedRenderer || ''))) {
    throw new Error(`CPU-only backend verification failed: ${JSON.stringify(info)}`);
  }
}

export function reserveOutput(out) {
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.mkdirSync(out); // Atomic no-clobber: never reuse an accepted or partial dataset.
  fs.writeFileSync(path.join(out, 'INCOMPLETE'), 'Render has not completed successfully.\n');
  fs.mkdirSync(path.join(out, 'images'));
  fs.mkdirSync(path.join(out, 'labels'));
}

export function assertStandardRendererRows(rows) {
  if (rows.some(row => ['ph_occlusion_low', 'ph_occlusion_low2', 'ph_occlusion_low3', 'ph_occlusion_h'].includes(row.experiment))) {
    throw new Error('occlusion experiments require render_occlusion_batch.mjs selection and pixel-proof QA');
  }
}

export async function serveRendering(directory) {
  const base = path.resolve(directory);
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.gltf': 'model/gltf+json', '.jpg': 'image/jpeg', '.png': 'image/png' };
  const server = http.createServer((request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const filename = path.resolve(base, '.' + pathname);
      if (!['GET', 'HEAD'].includes(request.method) || !filename.startsWith(base + path.sep)) {
        response.writeHead(403).end(); return;
      }
      if (!fs.statSync(filename).isFile()) { response.writeHead(404).end(); return; }
      response.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream' });
      if (request.method === 'HEAD') { response.end(); return; }
      fs.createReadStream(filename).on('error', () => response.destroy()).pipe(response);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    close: () => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifest = path.resolve(root, args.manifest || 'rendering/angles_manifest.jsonl');
  const out = path.resolve(root, args.out || 'data/synth/angles');
  const rendererFile = args.renderer || 'renderer.html';
  const engine = args.engine || 'three.js r128 procedural Staunton';
  const generator = args.generator || 'generate_angles.py';
  const backend = args.backend || (process.platform === 'darwin' ? 'metal' : 'swiftshader');
  const limit = Number(args.limit || 0);
  if (!Number.isInteger(limit) || limit < 0) throw new Error('limit must be a nonnegative integer');
  const rows = fs.readFileSync(manifest, 'utf8').trim().split('\n').map(JSON.parse).slice(0, limit || undefined);
  if (!rows.length || new Set(rows.map(row => row.id)).size !== rows.length ||
      rows.some(row => !/^\d{6}$/.test(row.id) || typeof row.fen !== 'string')) {
    throw new Error('Manifest must have unique six-digit IDs and FEN strings');
  }
  assertStandardRendererRows(rows);
  const sha256 = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const extraSources = (args.sources || '').split(',').filter(Boolean);
  const sourceFiles = [rendererFile, 'render_batch.mjs', generator, 'three.min.js', ...extraSources];
  const sourceSha256 = Object.fromEntries(sourceFiles.map(name => [name, sha256(path.join(root, 'rendering', name))]));
  sourceSha256[path.basename(manifest)] = sha256(manifest);
  const modulePath = args.playwright || process.env.PLAYWRIGHT_MODULE ||
    (process.platform === 'darwin' && fs.existsSync(macPlaywright) ? macPlaywright : 'playwright');
  const imported = await import(path.isAbsolute(modulePath) ? pathToFileURL(modulePath).href : modulePath);
  const { chromium } = imported;
  const executable = args.browser || process.env.CHROME_BIN ||
    (process.platform === 'darwin' && fs.existsSync(macChrome) ? macChrome : undefined);
  const options = launchOptions(backend, executable);
  reserveOutput(out);
  const browser = await chromium.launch(options);
  let server;
  try {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    page.setDefaultTimeout(120000);
    page.on('console', m => console.error('browser:', m.type(), m.text()));
    page.on('pageerror', e => console.error('browser pageerror:', e.message));
    page.on('requestfailed', request => console.error('asset request failed:', request.url(), request.failure()?.errorText));
    // Modern Linux Chromium restricts fetch() from file:// even for local glTF
    // assets. A private loopback origin avoids weakening browser security.
    if (backend === 'swiftshader') server = await serveRendering(path.join(root, 'rendering'));
    await page.goto(server ? `${server.url}/${rendererFile}` : pathToFileURL(path.join(root, 'rendering', rendererFile)).href);
    await page.waitForFunction(() => window.__rendererReady === true, null, { timeout: 120000 });
    const webgl = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      const gl = canvas && (canvas.getContext('webgl2') || canvas.getContext('webgl'));
      if (!gl) return { ok: false };
      const debug = gl.getExtension('WEBGL_debug_renderer_info');
      return {
        ok: true, version: gl.getParameter(gl.VERSION), renderer: gl.getParameter(gl.RENDERER),
        vendor: gl.getParameter(gl.VENDOR),
        unmaskedRenderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null,
        unmaskedVendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : null,
      };
    });
    checkBackend(backend, webgl);
    const runtime = {
      node: process.version, browser: browser.version(), playwright_module: modulePath,
      requested_backend: backend, launch_options: options, webgl,
    };
    fs.writeFileSync(path.join(out, 'runtime.json'), JSON.stringify(runtime, null, 2) + '\n');
    console.log('verified backend:', JSON.stringify(webgl));
    fs.writeFileSync(path.join(out, 'metadata.jsonl'), '');
    const t0 = Date.now();
    let n = 0;
    for (const row of rows) {
      const measured = await page.evaluate(r => window.renderConfig(r), row);
      const imagePath = path.join(out, 'images', row.id + '.png');
      await page.locator('canvas').screenshot({ path: imagePath, type: 'png' });
      fs.writeFileSync(path.join(out, 'labels', row.id + '.txt'), row.fen + '\n');
      fs.appendFileSync(path.join(out, 'metadata.jsonl'), JSON.stringify({
        ...row, image_sha256: sha256(imagePath), source_sha256: sourceSha256,
        renderer: { engine, three_revision: '128', backend, webgl_renderer: webgl.unmaskedRenderer, ...measured },
      }) + '\n');
      n++;
      if (n % 10 === 0) process.stdout.write(`rendered ${n}/${rows.length}\n`);
    }
    const seconds = (Date.now() - t0) / 1000;
    const summary = { rendered: n, seconds, images_per_second: n / seconds, source_sha256: sourceSha256, runtime };
    fs.writeFileSync(path.join(out, 'render_summary.json'), JSON.stringify(summary, null, 2) + '\n');
    console.log(JSON.stringify({ rendered: n, seconds, images_per_second: n / seconds, out }));
  } finally {
    try { await browser.close(); }
    finally { if (server) await server.close(); }
  }
  fs.unlinkSync(path.join(out, 'INCOMPLETE'));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
