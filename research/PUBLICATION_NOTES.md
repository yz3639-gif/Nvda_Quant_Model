# Publication metadata normalization — 2026-09-23

This publication makes the saved research evidence portable without running a new model or downloading newer market data. Source commits, engine hashes, configurations, recorded run dates, numerical outputs, and chart/report contents remain unchanged.

Personal absolute filesystem paths were replaced with repository-relative references. `legacy_workspace/` identifies the earlier source workspace; referenced files not included in this repository remain unavailable. Local branch labels were omitted. Historical module documentation now links to the current project overview and identifies its older "production baseline" terminology as historical.

Metadata normalization changes file bytes. Parent artifact maps, baseline file-size records, replay input references and stage checkpoints were updated from the published files, bottom-up. The original and published file hashes, original run identities, and numeric fingerprints are recorded in [PUBLICATION_PROVENANCE.json](PUBLICATION_PROVENANCE.json). Original metadata files also remain in a local, ignored backup under `research/runs/publication_backup/`.

[REPRODUCTION_CHECK.json](REPRODUCTION_CHECK.json) describes the original September 18 reproduction, not a claim that these normalized metadata files are byte-identical to the original export. No historical financial result was revised during publication. Original source/input identities remain in the publication provenance record; a refreshed baseline-directory hash identifies the normalized export.

Verify the published export with Python's standard library:

```bash
python scripts/verify_publication.py
```

The check verifies before/after numeric fingerprints, current artifact and checkpoint hash maps, baseline records, replay references, and retained source/input identities. Missing original market-data cache files are explicitly counted as unshipped inputs; this check does not imply a market-data redistribution license or independently reproduce their contents. The JSON fingerprint excludes fields named `bytes`, which are metadata file sizes; all other numeric leaves and all numeric CSV cells are compared, including their locations.
