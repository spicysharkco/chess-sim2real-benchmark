# Phuong2647 licence recovery: positive grant, exact version unresolved

**Result:** the bounded read-only recovery did not establish an asset-specific
CC BY version or legal-code text. The exact-version gate remains open. This is
**not** evidence that the asset has no licence or that permission never existed.

## Evidence recovered

1. The live [author asset page](https://blendswap.com/blend/29244) identifies
   Phuong2647 and labels this Chess set CC-BY.
2. Its public [download/login page](https://blendswap.com/blend/29244/download)
   states: “Free under CC-BY. Please credit Phuong2647 when you use it.” Its
   full-licence link points to site terms, not a versioned CC legal-code URL.
3. [BlendSwap's terms](https://blendswap.com/terms), section 6, defer to the licence
   attached to each upload. The site's BY-SA 3.0 USA statement does not identify
   this separately CC-BY-labelled model's version. Borrowing that version would
   also confuse two different licence types.
4. The [catalog API documentation](https://blendswap.com/3d-mcp-api/docs) exposes
   asset-level licence metadata and maps the current global `cc_by` enum to
   version 4.0. However, the anonymous asset-29244 endpoint returned **401**;
   no asset-level payload was retrieved. A current global enum is not used to
   backfill this historical model's grant. No key, account or credential was used.
5. Internet Archive CDX lookups for both the current and historical-looking asset
   paths returned **503**; the availability API returned **429**. No archived
   author snapshot was recovered in this attempt. These are access failures,
   not proof that an archive does not exist.

## Exact distribution checked

The [identified Castle Engine distribution](https://github.com/castle-engine/bad-chess/releases/tag/chess-models)
was fetched read-only: **2,247,473 bytes**, SHA-256
`1b7c9f03e0bfc9836e9c775bb04dda390cd1d860d5aeb31e7cef935cfcd2a9c4`.
It exactly matches the source archive already recorded for the frozen mesh.

Its 28 entries include 13 glTF files and a packed Blender file, but no README or
licence file. glTF asset headers provide exporter/version metadata, not a rights
notice. A read-only string scan of the Blender bytes found copyright/licence
schema-field names, not an actual licence value; Blender was not executed.
The [pinned redistributor AUTHORS record](https://github.com/castle-engine/bad-chess/blob/1edd1e4539931e6251e04bf55dd491dfe3762182/project/version_1_designed_in_editor/data/AUTHORS.md)
also identifies the model, author and CC-BY, without a version. The repository's
BSD code licence is explicitly not substituted for the 3D model licence.

URLs, retrieval status, snapshot hashes, archive member inventory and the short
redistributor notice are in [the evidence record](phuong_licence_recovery_2026-09-10/evidence.json).
Full fetched web pages and the source archive are retained only in the task's
private temporary evidence directories; they are not republished here.

## Implication for including phuong

Under the current **exact-version release gate**, do not approve redistribution
of the Phuong mesh/converted mesh or its rendered family on this evidence alone.
To include it, obtain asset-specific version/legal-code evidence, or an express
clarifying grant from the rights holder. This is the project's evidence gate,
not a blanket legal conclusion about an unversioned CC-BY notice.

Alternatively, a public data release can omit this family, but that requires a
**new release version and fresh manifests/receipts**, not deletion from a frozen
benchmark. C1 would lose 1,000 cell-images (800 dev, 200 final), leaving 19,000
cell-images with 19 factor cells; the shared 500 position groups remain. Removing
this geometry from the adaptation pool removes 1,000 groups / 2,000 images,
leaving 7,000 groups, so it cannot silently retain the registered 8,000-group claim.
No such filtering or rerendering was performed.

Existing aggregate evaluation results are not retroactively invalidated by this
evidence gap. Learned-weight distribution is a separate rights question, not
resolved merely by removing an asset from a package. The public-export scrub
failure is also independent and remains open. `release_ready=false`.
