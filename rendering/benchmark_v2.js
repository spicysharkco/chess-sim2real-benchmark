(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BenchmarkV2 = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';

  const PARTITIONS = Object.freeze({
    dev: 'benchmark_dev',
    final_test: 'benchmark_final_test',
    adaptation_train: 'adaptation_train',
  });
  const FACTOR_KEYS = Object.freeze([
    'elevation_band', 'framing', 'geometry', 'palette', 'parity',
  ]);
  const ELEVATION_BANDS = Object.freeze({
    '08_15': [8, 15], '15_25': [15, 25], '25_45': [25, 45], '45_60': [45, 60],
  });
  const FRAMING_BANDS = Object.freeze({compact: [.30, .40], standard: [.52, .61]});

  function fail(message) {
    throw new Error(`benchmark v2 row rejected: ${message}`);
  }

  function exactObject(value, keys, name) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${name} must be an object`);
    const actual = Object.keys(value).sort();
    const expected = [...keys].sort();
    if (JSON.stringify(actual) !== JSON.stringify(expected)) fail(`${name} keys differ from the contract`);
  }

  function deepEqual(left, right) {
    if (left === right) return true;
    if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false;
    if (Array.isArray(left) !== Array.isArray(right)) return false;
    if (Array.isArray(left)) {
      return left.length === right.length && left.every((value, index) => deepEqual(value, right[index]));
    }
    const leftKeys = Object.keys(left).sort();
    const rightKeys = Object.keys(right).sort();
    return deepEqual(leftKeys, rightKeys) && leftKeys.every(key => deepEqual(left[key], right[key]));
  }

  function rgb(value, name) {
    if (!Array.isArray(value) || value.length !== 3 ||
        value.some(channel => !Number.isInteger(channel) || channel < 0 || channel > 255)) {
      fail(`${name} must contain three integer RGB channels`);
    }
  }

  function material(value, name) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${name} must be an object`);
    rgb(value.rgb, `${name}.rgb`);
    for (const key of ['roughness', 'metalness']) {
      if (!Number.isFinite(value[key]) || value[key] < 0 || value[key] > 1) fail(`${name}.${key} must be in [0,1]`);
    }
  }

  function isRow(row) {
    return Boolean(row && (row.benchmark_schema === 2 || row.experiment === 'benchmark_v2'));
  }

  function validateRow(row, expectedGeometry, expectedFamily) {
    if (!row || row.benchmark_schema !== 2 || row.experiment !== 'benchmark_v2') {
      fail('schema and experiment must identify benchmark_v2');
    }
    if (!Object.prototype.hasOwnProperty.call(PARTITIONS, row.partition)) fail('unknown partition');
    if (row.data_usage !== PARTITIONS[row.partition]) fail('partition and data_usage disagree');
    if (typeof row.position_group !== 'string' || !/^[0-9a-f]{64}$/.test(row.position_group)) {
      fail('position_group must be a lowercase SHA-256 hex digest');
    }
    exactObject(row.factors, FACTOR_KEYS, 'factors');
    for (const key of FACTOR_KEYS) {
      if (typeof row.factors[key] !== 'string' || !row.factors[key]) fail(`factors.${key} must be a non-empty string`);
    }
    if (row.factors.geometry !== expectedFamily) fail('geometry factor does not match renderer');
    if (!['a1_dark', 'a1_light'].includes(row.factors.parity) || row.parity !== row.factors.parity) {
      fail('top-level parity must equal the declared parity factor');
    }
    if (!deepEqual(row.piece_geometry, expectedGeometry)) fail('piece_geometry differs from the pinned descriptor');
    if (!row.board_palette || typeof row.board_palette !== 'object' || Array.isArray(row.board_palette)) {
      fail('board_palette must be an object');
    }
    rgb(row.board_palette.light_rgb, 'board_palette.light_rgb');
    rgb(row.board_palette.dark_rgb, 'board_palette.dark_rgb');
    if (!row.piece_palette || typeof row.piece_palette !== 'object' || Array.isArray(row.piece_palette)) {
      fail('piece_palette must be an object');
    }
    material(row.piece_palette.white, 'piece_palette.white');
    material(row.piece_palette.black, 'piece_palette.black');
    if (!row.camera || typeof row.camera !== 'object' || Array.isArray(row.camera)) fail('camera must be an object');
    for (const key of ['azimuth_deg', 'elevation_deg', 'roll_deg', 'target_board_width_fraction']) {
      if (!Number.isFinite(row.camera[key])) fail(`camera.${key} must be finite`);
    }
    if (row.camera.target_board_width_fraction <= 0 || row.camera.target_board_width_fraction >= 1) {
      fail('camera.target_board_width_fraction must be in (0,1)');
    }
    const elevationRange = ELEVATION_BANDS[row.factors.elevation_band];
    const framingRange = FRAMING_BANDS[row.factors.framing];
    if (!elevationRange || row.camera.elevation_deg < elevationRange[0] || row.camera.elevation_deg > elevationRange[1]) {
      fail('camera elevation disagrees with its factor band');
    }
    if (!framingRange || row.camera.target_board_width_fraction < framingRange[0] ||
        row.camera.target_board_width_fraction > framingRange[1]) {
      fail('camera framing disagrees with its factor band');
    }
    return row;
  }

  function squareRole(row, rankIndex, fileIndex) {
    if (!Number.isInteger(rankIndex) || rankIndex < 0 || rankIndex > 7 ||
        !Number.isInteger(fileIndex) || fileIndex < 0 || fileIndex > 7) fail('square index outside board');
    const odd = (rankIndex + fileIndex) % 2 === 1;
    return row.parity === 'a1_light' ? (odd ? 'light' : 'dark') : (odd ? 'dark' : 'light');
  }

  function linearColor(THREE, value) {
    return new THREE.Color().setRGB(...value.map(channel => channel / 255)).convertSRGBToLinear();
  }

  function boardMaterials(THREE, row) {
    return {
      light: new THREE.MeshBasicMaterial({color: linearColor(THREE, row.board_palette.light_rgb)}),
      dark: new THREE.MeshBasicMaterial({color: linearColor(THREE, row.board_palette.dark_rgb)}),
    };
  }

  function pieceMaterials(THREE, row) {
    return Object.fromEntries(['white', 'black'].map(side => {
      const spec = row.piece_palette[side];
      return [side, new THREE.MeshStandardMaterial({
        color: linearColor(THREE, spec.rgb), roughness: spec.roughness, metalness: spec.metalness,
      })];
    }));
  }

  function recolor(object, materials, side) {
    object.traverse(child => { if (child.isMesh) child.material = materials[side]; });
  }

  function configureCamera(THREE, camera, _sceneRoot, cfg, width, height) {
    camera.updateProjectionMatrix();
    const azimuth = cfg.azimuth_deg * Math.PI / 180;
    const elevation = cfg.elevation_deg * Math.PI / 180;
    const actualRoll = cfg.roll_deg;
    let distance = 17;
    function setAt() {
      const horizontal = distance * Math.cos(elevation);
      camera.position.set(horizontal * Math.sin(azimuth), distance * Math.sin(elevation),
                          horizontal * Math.cos(azimuth));
      camera.up.set(0, 1, 0);
      camera.lookAt(0, .28, 0);
      camera.rotateZ(actualRoll * Math.PI / 180);
      camera.updateMatrixWorld(true);
    }
    const edge = 4.003;
    const corners = [];
    for (const y of [0, -.10]) for (const x of [-edge, edge]) for (const z of [-edge, edge]) corners.push([x, y, z]);
    function projected() {
      setAt();
      return corners.map(value => new THREE.Vector3(...value).project(camera));
    }
    for (let index = 0; index < (cfg.target_board_width_fraction >= .52 ? 16 : 8); index++) {
      const points = projected();
      const xs = points.map(point => point.x), ys = points.map(point => point.y);
      const widthFraction = (Math.max(...xs) - Math.min(...xs)) / 2;
      const heightFraction = (Math.max(...ys) - Math.min(...ys)) / 2;
      const maxX = Math.max(...xs.map(Math.abs)), maxY = Math.max(...ys.map(Math.abs));
      distance *= cfg.target_board_width_fraction >= .52 ? widthFraction / cfg.target_board_width_fraction :
        Math.max(widthFraction / cfg.target_board_width_fraction, heightFraction / .78,
                 maxX / (1 - 40 / width), maxY / (1 - 40 / height));
    }
    // V2 centres the fixed board slab, never the pieces. Thus geometry and FEN
    // cannot change the camera within a paired factor group.
    const unshifted = projected();
    const unshiftedY = unshifted.map(point => point.y);
    const imageOffsetY = (Math.min(...unshiftedY) + Math.max(...unshiftedY)) * height / 4;
    camera.projectionMatrix.elements[9] += 2 * imageOffsetY / height;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    const points = projected();
    const xs = points.map(point => (point.x + 1) / 2), ys = points.map(point => (1 - point.y) / 2);
    const bounds = {left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys)};
    const projectedWidth = bounds.right - bounds.left;
    if (cfg.target_board_width_fraction >= .52 &&
        Math.abs(projectedWidth - cfg.target_board_width_fraction) > 2 / width + 1e-6) {
      fail('projected board width differs from requested target');
    }
    const canonicalCornersPx = [
      [-4, 0, -4], [4, 0, -4], [4, 0, 4], [-4, 0, 4],
    ].map(value => {
      const point = new THREE.Vector3(...value).project(camera);
      return [(point.x + 1) * width / 2, (1 - point.y) * height / 2];
    });
    return {
      distance, actual_roll_deg: actualRoll, projected_board_bounds: bounds,
      image_offset_y_px: imageOffsetY,
      canonical_corner_names: ['a8', 'h8', 'h1', 'a1'],
      canonical_corners_px: canonicalCornersPx,
      projected_board_width_fraction: projectedWidth,
      projected_board_height_fraction: bounds.bottom - bounds.top,
    };
  }

  async function loadGeometry(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`pinned geometry descriptor fetch failed: ${response.status}`);
    return response.json();
  }

  return Object.freeze({
    PARTITIONS, boardMaterials, configureCamera, deepEqual, isRow, loadGeometry, pieceMaterials, recolor,
    squareRole, validateRow,
  });
});
