"""Configuration for in-memory checkpointing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InMemoryCkptConfig:
    """Configuration for InMemoryCheckpointManager.

    Args:
        enable: Whether in-memory checkpointing is active.
        interval: Save an in-memory checkpoint every N training steps.
        replication_jump: Spacing between paired ranks for P2P replication.
            -1 means auto-detect (torch.cuda.device_count(), i.e., one node width).
        replication_chunk_size: Bytes per replication send. Controls the maximum
            head-of-line blocking delay on FSDP all-gather:
            max_ag_delay = chunk_size / nic_bandwidth.
            Use estimate_chunk_size() to compute from your model's layer compute time.
            Default 16 MiB → 320 μs max delay at 400 Gbps (3.8% of 8.4ms buffer).
        disk_flush_interval: Flush to disk every N steps. 0 disables disk flush.
        disk_folder: Directory for the node-local disk tier — fast reload source
            for a same-node restart. GEMINI is single-tier by design: durable /
            global checkpointing is the pre-training framework's responsibility,
            reached via the caller's load_fallback when GEMINI has nothing to load.
        verify_integrity: If True, store a per-shard CRC-32 on every disk write and
            verify it on load. Guards at-rest / in-flight byte corruption (DRAM
            rot, transfer flips, NVMe errors) — the failure mode layer replay
            cannot see. A shard that fails its checksum is treated as absent.
        skip_replication_if_hsdp: Skip P2P replication when HSDP provides natural replicas.
        pin_memory: Use pinned (page-locked) CPU memory for async GPU→CPU copy.
    """

    enable: bool = True
    interval: int = 10
    replication_jump: int = -1
    replication_chunk_size: int = 16 * 1024 * 1024  # 16 MiB
    disk_flush_interval: int = 100
    disk_folder: str = "./checkpoints"
    verify_integrity: bool = False
    skip_replication_if_hsdp: bool = True
    pin_memory: bool = True
