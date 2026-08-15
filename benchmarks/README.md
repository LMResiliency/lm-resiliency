# Healthy-Path Benchmarks

The native-PyTorch benchmark runs baseline, GEMINI-only, SCOUT-only, and combined modes in fresh `torchrun` processes. It records the exact revision, Python/PyTorch/CUDA environment, topology dimensions, workload and seed, checkpoint size and cadence, pinning and replication settings, replay cadence, throughput, p50/p95 step latency, and peak host/GPU memory.

Run a two-GPU comparison:

```bash
python benchmarks/run_healthy_path.py \
  --device cuda \
  --world-size 2 \
  --output-dir /tmp/lm-resiliency-benchmark \
  --enforce
```

The controller writes one JSON record per mode plus `summary.json`, `summary.md`, `commands.json`, and SHA-256 checksums. Thresholds live in [thresholds.json](thresholds.json). They are stability alarms for this small representative workload, not replacements for the historical workload-specific percentages in the project documentation. The two-peer SCOUT cases measure healthy-path cost but do not qualify exact fault attribution, which requires at least three equivalent peers.

For a fast CPU harness smoke without a performance claim:

```bash
python benchmarks/run_healthy_path.py \
  --device cpu \
  --world-size 1 \
  --modes baseline gemini \
  --steps 3 \
  --warmup-steps 1 \
  --hidden-size 32 \
  --layers 2 \
  --heads 4 \
  --sequence-length 8 \
  --batch-size 2 \
  --output-dir /tmp/lm-resiliency-benchmark-smoke
```

Do not compare results across different hardware, framework versions, workload shapes, or topology records. Scheduled regression decisions compare all modes within one isolated run on the same runner.
