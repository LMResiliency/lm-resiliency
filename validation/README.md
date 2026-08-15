# Validation Evidence

Every published qualification bundle uses the versioned
[`evidence-schema-v1.json`](evidence-schema-v1.json) contract:

```text
<campaign>/
  manifest.json
  summary.json
  environment.json
  commands.txt
  checksums.txt
  <logs and raw results>
```

`manifest.json` binds the campaign to a full Git commit, package version, ref, tier,
command IDs, seed and configuration, framework scope, hardware topology, artifact
location, qualification boundaries, and every payload digest. `summary.json` records
aggregate counts, metrics, and per-command results. `environment.json` records the
runner hardware and software versions. `commands.txt` is the exact replayable command
inventory. `checksums.txt` covers the manifest and every payload file.

The executable schema validator intentionally uses only the Python standard library:

```bash
python scripts/validation_evidence.py verify \
  --bundle /path/to/evidence \
  --expected-commit "$(git rev-parse HEAD)"
```

The `--expected-commit` check fails with an explicit `validation evidence is stale`
message whenever a bundle covers another revision. A green result is current only when
that comparison succeeds. The artifact name includes the same full commit SHA, and its
manifest links back to the immutable workflow run that produced it.

The primary CPU matrix and trusted single-host GPU workflow currently publish sealed
bundles. Use `seal` for single-GPU, multi-host, TorchTitan, Megatron Core, DeepSpeed,
GEMINI, SCOUT, or MoE campaigns by first producing the three payload files. Record each
framework actually exercised and state topology and coverage limitations as explicit
`--boundary` values. Large logs may remain in the workflow or release artifact; their
path, size, and digest must still appear in the bundle manifest.

Do not copy a passing status between revisions. Re-run the campaign and publish a new
SHA-named bundle instead.
