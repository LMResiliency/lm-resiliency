"""C³: Consensus Collective Communication.

C³ gathers compact comparable evidence and returns an explicit verdict,
outlier bitmap, and the ordered peer evidence used to derive both.
"""

from __future__ import annotations

import enum
import logging
import math
import struct
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

SIGNATURE_OFFSET_BASIS = 14695981039346656037
SIGNATURE_PRIME = 1099511628211
SIGNATURE_MASK = (1 << 64) - 1
SIGNED_INT64_MAX = (1 << 63) - 1

NUM_CHUNKS = 512


class C3Mode(enum.Enum):
    EXACT = "exact"
    STATISTICAL = "statistical"


class C3Status(str, enum.Enum):
    """Consensus verdict for one ordered peer group."""

    AGREE = "agree"
    ATTRIBUTED = "attributed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class C3Result:
    """Status, outlier bitmap, and gathered evidence for one C³ call."""

    status: C3Status
    bitmap: list[int]
    evidence: list[int | float]


def _xor_fold_to_blocks(flat_int64: torch.Tensor) -> torch.Tensor:
    """Fold a flat int64 tensor down to NUM_CHUNKS values via salted pairwise XOR.

    Each folding level XORs the right half with a level-dependent salt before
    combining with the left half. This makes the reduction position-sensitive:
    swapping or identically corrupting elements at mirror positions no longer
    cancels out, because the salt differentiates folding levels.

    The input is zero-padded to equal-width chunks, then each odd folding level
    gets one zero column. Padding is essential: truncating to equal chunks or
    halving an odd width would leave some input elements out of the signature.

    Works on both CPU and CUDA tensors — all ops are vectorized torch ops.
    Returns a 1-D tensor of at most NUM_CHUNKS values on the same device.
    """
    num_elements = flat_int64.numel()

    if num_elements <= NUM_CHUNKS:
        return flat_int64

    fold_width = (num_elements + NUM_CHUNKS - 1) // NUM_CHUNKS
    padded_elements = NUM_CHUNKS * fold_width
    if padded_elements > num_elements:
        flat_int64 = torch.nn.functional.pad(flat_int64, (0, padded_elements - num_elements))
    chunks = flat_int64.view(NUM_CHUNKS, fold_width)

    # Repeatedly halve columns via salted XOR until one column remains.
    # Salt = a different large odd constant per level, preventing cancellation
    # across mirror positions.
    data = chunks
    level = 0
    while data.shape[1] > 1:
        if data.shape[1] % 2:
            data = torch.nn.functional.pad(data, (0, 1))
        half = data.shape[1] // 2
        left = data[:, :half]
        right = data[:, half:]
        # Level-dependent salt: multiply by a different prime each level
        salt = torch.tensor(
            SIGNATURE_PRIME * (level + 1) & SIGNATURE_MASK,
            dtype=torch.int64,
            device=data.device,
        )
        data = left ^ (right * salt)
        level += 1

    return data[:, 0]


def _finalize_signature(block_values: list[int], payload_nbytes: int) -> int:
    """Sequential XOR-multiply chain over block digests. Returns signed int64."""
    h = SIGNATURE_OFFSET_BASIS
    for val in block_values:
        h = (h ^ (val & SIGNATURE_MASK)) & SIGNATURE_MASK
        h = (h * SIGNATURE_PRIME) & SIGNATURE_MASK
    # Distinguish real trailing zero bytes from zero padding added for folding.
    h = (h ^ (payload_nbytes & SIGNATURE_MASK)) & SIGNATURE_MASK
    h = (h * SIGNATURE_PRIME) & SIGNATURE_MASK
    if h > SIGNED_INT64_MAX:
        h = h - (1 << 64)
    return h


def _to_int64_view(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a tensor as int64, padding to 8-byte alignment if needed."""
    flat = tensor.detach().contiguous().view(-1)
    if flat.dtype == torch.int64:
        return flat
    flat = flat.view(torch.uint8)
    remainder = flat.numel() % 8
    if remainder != 0:
        flat = torch.nn.functional.pad(flat, (0, 8 - remainder))
    return flat.view(torch.int64)


def _compute_tensor_signature(tensor: torch.Tensor) -> int:
    """Compute a metadata-aware 64-bit tensor signature.

    Works on both CPU and CUDA tensors. The bulk XOR-fold runs on whatever
    device the tensor lives on; only the final 512 block values are moved
    to CPU for the sequential chain (unavoidable without a custom CUDA kernel).
    """
    flat = _to_int64_view(tensor)
    block_xors = _xor_fold_to_blocks(flat)

    # Move at most 512 block digests (4 KB) to CPU.
    block_values = block_xors.cpu().tolist() if block_xors.is_cuda else block_xors.tolist()
    payload_nbytes = tensor.numel() * tensor.element_size()
    content_signature = _finalize_signature(block_values, payload_nbytes)
    dtype_value = _signature_string(str(tensor.dtype))
    metadata = [content_signature, tensor.ndim, *tensor.shape, dtype_value]
    return _finalize_signature(metadata, payload_nbytes)


def _signature_string(value: str) -> int:
    signature = 0
    for byte in value.encode("ascii"):
        signature = ((signature ^ byte) * SIGNATURE_PRIME) & SIGNATURE_MASK
    return signature


def _signature_bytes(value: bytes) -> int:
    block_values = [
        int.from_bytes(value[offset : offset + 8].ljust(8, b"\0"), "little")
        for offset in range(0, len(value), 8)
    ]
    return _finalize_signature(block_values, len(value))


def _compute_tensor_sequence_signature(tensors: Sequence[torch.Tensor]) -> int:
    """Hash a structured tensor payload without concatenating its tensor leaves."""
    values: list[int] = [len(tensors)]
    total_nbytes = 0
    for tensor in tensors:
        local = tensor.to_local() if type(tensor).__name__ == "DTensor" else tensor
        values.append(_compute_tensor_signature(local))
        values.append(local.ndim)
        values.extend(local.shape)
        values.append(_signature_string(str(local.dtype)))
        total_nbytes += local.numel() * local.element_size()
    return _finalize_signature(values, total_nbytes)


def _compute_structure_signature(value: Any) -> int:
    """Hash nested replay metadata without serializing or copying full CUDA tensors."""
    values: list[int] = []
    payload_nbytes = 0

    def append(item: Any) -> None:
        nonlocal payload_nbytes
        type_tag = f"{type(item).__module__}.{type(item).__qualname__}"
        values.append(_signature_string(type_tag))

        if item is None:
            return
        if isinstance(item, torch.Tensor):
            local = item.to_local() if type(item).__name__ == "DTensor" else item
            values.append(_compute_tensor_signature(local))
            payload_nbytes += local.numel() * local.element_size()
            return
        if isinstance(item, bool):
            values.append(int(item))
            payload_nbytes += 1
            return
        if isinstance(item, int):
            encoded = str(item).encode("ascii")
            values.append(_signature_bytes(encoded))
            payload_nbytes += len(encoded)
            return
        if isinstance(item, float):
            encoded = struct.pack("!d", item)
            values.append(_signature_bytes(encoded))
            payload_nbytes += len(encoded)
            return
        if isinstance(item, enum.Enum):
            append(item.value)
            return
        if isinstance(item, (torch.dtype, torch.device)):
            encoded = str(item).encode("ascii")
            values.append(_signature_bytes(encoded))
            payload_nbytes += len(encoded)
            return
        if isinstance(item, type):
            encoded = f"{item.__module__}.{item.__qualname__}".encode("utf-8")
            values.append(_signature_bytes(encoded))
            payload_nbytes += len(encoded)
            return
        if isinstance(item, (str, Path)):
            encoded = str(item).encode("utf-8")
            values.append(_signature_bytes(encoded))
            payload_nbytes += len(encoded)
            return
        if isinstance(item, bytes):
            values.append(_signature_bytes(item))
            payload_nbytes += len(item)
            return
        if isinstance(item, Mapping):
            entries = sorted(item.items(), key=lambda pair: repr(pair[0]))
            values.append(len(entries))
            for key, nested in entries:
                append(key)
                append(nested)
            return
        if isinstance(item, (Sequence, Set)) and not isinstance(item, (str, bytes)):
            entries = sorted(item, key=repr) if isinstance(item, Set) else item
            values.append(len(entries))
            for nested in entries:
                append(nested)
            return

        try:
            import numpy as np

            if isinstance(item, np.ndarray):
                append(torch.from_numpy(item))
                return
            if isinstance(item, np.generic):
                append(item.item())
                return
        except ImportError:
            pass

        raise TypeError(f"unsupported C3 structured value: {type(item).__qualname__}")

    append(value)
    return _finalize_signature(values, payload_nbytes)


class C3:
    """Consensus Collective Communication primitive.

    Identifies outlier ranks within a peer group. Independent of training
    communication — uses its own process group.

    Tensor payloads are reduced to metadata-aware 64-bit signatures before
    AllGather, so diagnostic traffic does not scale with payload size.

    Args:
        group: Process group for C³ collectives. If None, uses WORLD.
        nccl_group: Separate NCCL group for GPU signature AllGather.
            If None and the input is a CUDA tensor, uses `group` (which
            must support NCCL ops). For CPU-only usage, leave as None.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        nccl_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._group = group
        self._nccl_group = nccl_group
        self._rank = dist.get_rank(group) if group else dist.get_rank()
        self._world_size = dist.get_world_size(group) if group else dist.get_world_size()

    @property
    def group_size(self) -> int:
        return self._world_size

    def run_scalar(
        self,
        value: int | float,
        mode: C3Mode = C3Mode.EXACT,
        threshold_sigma: float = 3.0,
    ) -> C3Result:
        """Direct path: run C³ on a scalar payload.

        AllGathers the scalar from all ranks, then locally identifies outliers.

        Args:
            value: Local diagnostic payload (e.g., op_id, timing).
            mode: EXACT for majority vote, STATISTICAL for z-score outlier.
            threshold_sigma: Z-score threshold for statistical mode.

        Returns:
            Status, N-element bitmap, and ordered gathered values.
        """
        if isinstance(value, int):
            local_tensor = torch.tensor([value], dtype=torch.int64)
        else:
            local_tensor = torch.tensor([value], dtype=torch.float64)

        gathered = [torch.zeros_like(local_tensor) for _ in range(self._world_size)]
        dist.all_gather(gathered, local_tensor, group=self._group)

        values = [t.item() for t in gathered]
        return self.classify_evidence(values, mode, threshold_sigma)

    def run_tensor_sequence(
        self,
        tensors: Sequence[torch.Tensor],
    ) -> C3Result:
        """Compare a structured tensor payload using one aggregate signature."""
        signature = _compute_tensor_sequence_signature(tensors)
        return self.run_scalar(signature, mode=C3Mode.EXACT)

    def run_structure(self, value: Any) -> C3Result:
        """Compare a nested replay input or RNG snapshot exactly."""
        return self.run_scalar(_compute_structure_signature(value), mode=C3Mode.EXACT)

    def run_tensor_groups(
        self,
        groups: Mapping[str, Sequence[torch.Tensor]],
    ) -> dict[str, C3Result]:
        """Compare several tensor payloads with one signature AllGather.

        Each group retains an independent consensus result, which lets callers
        distinguish output, input-gradient, and parameter-gradient divergence
        without concatenating their potentially large tensors.
        """
        names = list(groups)
        signatures = torch.tensor(
            [_compute_tensor_sequence_signature(groups[name]) for name in names],
            dtype=torch.int64,
        )
        gathered = [torch.zeros_like(signatures) for _ in range(self._world_size)]
        dist.all_gather(gathered, signatures, group=self._group)
        return {
            name: self.classify_evidence(
                [int(peer[index].item()) for peer in gathered],
                C3Mode.EXACT,
                threshold_sigma=0.0,
            )
            for index, name in enumerate(names)
        }

    def run_tensor(
        self,
        tensor: torch.Tensor,
        mode: C3Mode = C3Mode.EXACT,
        threshold_sigma: float = 3.0,
    ) -> C3Result:
        """Run C³ on a tensor through a compact signature AllGather.

        Args:
            tensor: Local diagnostic payload on CPU or CUDA.
            mode: EXACT for bitwise comparison, STATISTICAL for value comparison.
            threshold_sigma: Z-score threshold for statistical mode.

        Returns:
            Status, outlier bitmap, and ordered gathered signatures.
        """
        if tensor.is_cuda:
            return self._run_tensor_gpu(tensor, mode, threshold_sigma)
        else:
            return self._run_tensor_cpu(tensor, mode, threshold_sigma)

    def _run_tensor_gpu(
        self,
        tensor: torch.Tensor,
        mode: C3Mode,
        threshold_sigma: float,
    ) -> C3Result:
        """GPU path: signature computed on-device and AllGathered via NCCL.

        No D2H copy of the full tensor. The local fold runs on GPU, producing
        512 block digests that are copied to CPU (4 KB) for the sequential
        chain. The final 8-byte signature is placed on GPU for NCCL AllGather.
        """
        N = self._world_size
        _ = mode  # GPU path always uses signature comparison (exact match)

        sig = _compute_tensor_signature(tensor)
        sig_tensor = torch.tensor([sig], dtype=torch.int64, device=tensor.device)

        # AllGather signatures on GPU via NCCL (8 bytes × N — negligible)
        all_sigs = [torch.zeros(1, dtype=torch.int64, device=tensor.device) for _ in range(N)]
        tensor_group = self._nccl_group if self._nccl_group is not None else self._group
        dist.all_gather(all_sigs, sig_tensor, group=tensor_group)

        sig_values = [t.item() for t in all_sigs]
        return self.classify_evidence(sig_values, C3Mode.EXACT, threshold_sigma)

    def _run_tensor_cpu(
        self,
        tensor: torch.Tensor,
        mode: C3Mode,
        threshold_sigma: float,
    ) -> C3Result:
        """CPU path: signature AllGather without an ambiguous XOR shortcut."""
        N = self._world_size
        sig = _compute_tensor_signature(tensor)
        sig_tensor = torch.tensor([sig], dtype=torch.int64)
        all_sigs = [torch.zeros(1, dtype=torch.int64) for _ in range(N)]
        dist.all_gather(all_sigs, sig_tensor, group=self._group)

        sig_values = [t.item() for t in all_sigs]
        return self.classify_evidence(sig_values, C3Mode.EXACT, threshold_sigma)

    @staticmethod
    def classify_evidence(
        values: list[int | float],
        mode: C3Mode,
        threshold_sigma: float = 3.0,
    ) -> C3Result:
        """Apply the deterministic paper verdict rule to gathered evidence."""
        N = len(values)
        if N == 0:
            raise ValueError("C3 evidence must contain at least one peer value")
        if mode is C3Mode.STATISTICAL and threshold_sigma <= 0:
            raise ValueError("statistical C3 threshold_sigma must be positive")
        bitmap = [0] * N

        if mode == C3Mode.EXACT:
            counter = Counter(values)
            if len(counter) <= 1:
                return C3Result(C3Status.AGREE, bitmap, list(values))
            majority_value, majority_count = counter.most_common(1)[0]
            if majority_count <= N // 2:
                return C3Result(C3Status.INCONCLUSIVE, bitmap, list(values))
            for i, v in enumerate(values):
                if v != majority_value:
                    bitmap[i] = 1
            return C3Result(C3Status.ATTRIBUTED, bitmap, list(values))

        if N <= 1:
            return C3Result(C3Status.AGREE, bitmap, list(values))

        fvalues = [float(v) for v in values]
        sorted_vals = sorted(fvalues)
        median = sorted_vals[N // 2]

        scale = _robust_scale(sorted_vals, median, threshold_sigma)
        if scale is None:
            return C3Result(C3Status.AGREE, bitmap, list(values))

        for i, v in enumerate(fvalues):
            if abs(v - median) > threshold_sigma * scale:
                bitmap[i] = 1
        status = C3Status.ATTRIBUTED if any(bitmap) else C3Status.AGREE
        return C3Result(status, bitmap, list(values))


def _robust_scale(sorted_vals: list[float], median: float, threshold_sigma: float) -> float | None:
    """Compute robust dispersion: MAD → IQR → range fallback.

    Returns None if all values are effectively identical.
    """
    N = len(sorted_vals)
    abs_devs = sorted(abs(v - median) for v in sorted_vals)
    mad = abs_devs[N // 2]

    if mad >= 1e-12:
        return mad

    q1 = _linear_percentile(sorted_vals, 0.25)
    q3 = _linear_percentile(sorted_vals, 0.75)
    iqr = q3 - q1
    if iqr >= 1e-12:
        return iqr

    value_range = sorted_vals[-1] - sorted_vals[0]
    if value_range < 1e-9:
        return None

    return value_range / (2 * threshold_sigma)


def _linear_percentile(sorted_vals: list[float], quantile: float) -> float:
    position = (len(sorted_vals) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_vals[lower]
    weight = position - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
