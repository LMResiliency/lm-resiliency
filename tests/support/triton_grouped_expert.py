"""Instrumented Triton grouped-GEMM backend for SCOUT qualification.

The persistent scheduling structure follows Triton's MIT-licensed grouped-GEMM
tutorial. The kernels here add masked tails, backward computation, logical-work
tracing, and deterministic fault injection for validation.

Upstream source:
https://github.com/triton-lang/triton/blob/v3.7.1/python/tutorials/08-grouped-gemm.py
Triton revision: f797708c0626e5f9840ca5b0a98790e2c7cb09ad

Copyright 2018-2020 Philippe Tillet
Copyright 2020-2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl

from lm_resiliency.detection.moe_regimes import CTASemantics, ExecutionHints

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_NUM_WARPS = 4
_NUM_STAGES = 3
_TRACE_FIELDS = 6

_M_TAIL = 1
_N_TAIL = 2
_FIRST_IN_EXPERT = 4
_LAST_IN_EXPERT = 8
_QUEUE_REPEAT = 16
_REDUCTION_TAIL = 32
_FIRST_REDUCTION = 64
_LAST_REDUCTION = 128


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@triton.jit
def _persistent_grouped_linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    counts_ptr,
    trace_ptr,
    fault_work_item,
    REDUCTION: tl.constexpr,
    OUTPUT: tl.constexpr,
    WEIGHT_EXPERT_STRIDE: tl.constexpr,
    WEIGHT_REDUCTION_STRIDE: tl.constexpr,
    WEIGHT_OUTPUT_STRIDE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    row_start = 0
    problem_start = 0
    num_n_tiles = tl.cdiv(OUTPUT, BLOCK_N)
    for expert in range(NUM_EXPERTS):
        rows = tl.load(counts_ptr + expert)
        num_m_tiles = tl.cdiv(rows, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        while tile_idx >= problem_start and tile_idx < problem_start + num_tiles:
            tile_in_expert = tile_idx - problem_start
            tile_m = tile_in_expert // num_n_tiles
            tile_n = tile_in_expert % num_n_tiles

            offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offsets_k = tl.arange(0, BLOCK_K)
            input_offsets = (row_start + offsets_m[:, None]) * REDUCTION + offsets_k[None, :]
            weight_offsets = (
                expert * WEIGHT_EXPERT_STRIDE
                + offsets_k[:, None] * WEIGHT_REDUCTION_STRIDE
                + offsets_n[None, :] * WEIGHT_OUTPUT_STRIDE
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for reduction_block in range(0, tl.cdiv(REDUCTION, BLOCK_K)):
                reduction_offsets = reduction_block * BLOCK_K + offsets_k
                values = tl.load(
                    input_ptr + input_offsets,
                    mask=(offsets_m[:, None] < rows) & (reduction_offsets[None, :] < REDUCTION),
                    other=0.0,
                )
                weights = tl.load(
                    weight_ptr + weight_offsets,
                    mask=(reduction_offsets[:, None] < REDUCTION) & (offsets_n[None, :] < OUTPUT),
                    other=0.0,
                )
                accumulator += tl.dot(values, weights)
                input_offsets += BLOCK_K
                weight_offsets += BLOCK_K * WEIGHT_REDUCTION_STRIDE

            accumulator += tl.where(tile_idx == fault_work_item, 1.0, 0.0)
            output_offsets = (row_start + offsets_m[:, None]) * OUTPUT + offsets_n[None, :]
            tl.store(
                output_ptr + output_offsets,
                accumulator,
                mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < OUTPUT),
            )

            m_tail = (tile_m == num_m_tiles - 1) & (rows % BLOCK_M != 0)
            n_tail = (tile_n == num_n_tiles - 1) & (OUTPUT % BLOCK_N != 0)
            first = tile_in_expert == 0
            last = tile_in_expert == num_tiles - 1
            repeated = tile_idx >= NUM_SM
            role_code = m_tail + n_tail * 2 + first * 4 + last * 8 + repeated * 16
            trace_offset = tile_idx * 6
            tl.store(trace_ptr + trace_offset, expert)
            tl.store(trace_ptr + trace_offset + 1, tile_m)
            tl.store(trace_ptr + trace_offset + 2, tile_n)
            tl.store(trace_ptr + trace_offset + 3, tile_idx // NUM_SM)
            tl.store(trace_ptr + trace_offset + 4, role_code)
            tl.store(trace_ptr + trace_offset + 5, tl.program_id(0))

            tile_idx += NUM_SM
        row_start += rows
        problem_start += num_tiles


@triton.jit
def _persistent_grouped_weight_grad_kernel(
    input_ptr,
    grad_output_ptr,
    grad_weight_ptr,
    counts_ptr,
    trace_ptr,
    fault_work_item,
    HIDDEN: tl.constexpr,
    OUTPUT: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    tiles_m = tl.cdiv(HIDDEN, BLOCK_M)
    tiles_n = tl.cdiv(OUTPUT, BLOCK_N)
    tiles_per_expert = tiles_m * tiles_n
    row_start = 0
    logical_start = 0
    for expert in range(NUM_EXPERTS):
        rows = tl.load(counts_ptr + expert)
        reduction_chunks = tl.cdiv(rows, BLOCK_K)
        expert_tile_start = expert * tiles_per_expert
        while tile_idx >= expert_tile_start and tile_idx < expert_tile_start + tiles_per_expert:
            tile_in_expert = tile_idx - expert_tile_start
            tile_m = tile_in_expert // tiles_n
            tile_n = tile_in_expert % tiles_n
            offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offsets_k = tl.arange(0, BLOCK_K)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            reduction_chunk = 0
            while reduction_chunk < reduction_chunks:
                reduction_offsets = reduction_chunk * BLOCK_K + offsets_k
                input_offsets = (row_start + reduction_offsets[:, None]) * HIDDEN + offsets_m[
                    None, :
                ]
                grad_offsets = (row_start + reduction_offsets[:, None]) * OUTPUT + offsets_n[
                    None, :
                ]
                values = tl.load(
                    input_ptr + input_offsets,
                    mask=(reduction_offsets[:, None] < rows) & (offsets_m[None, :] < HIDDEN),
                    other=0.0,
                )
                gradients = tl.load(
                    grad_output_ptr + grad_offsets,
                    mask=(reduction_offsets[:, None] < rows) & (offsets_n[None, :] < OUTPUT),
                    other=0.0,
                )
                accumulator += tl.dot(tl.trans(values), gradients)

                logical_idx = logical_start + tile_in_expert * reduction_chunks + reduction_chunk
                accumulator += tl.where(logical_idx == fault_work_item, 1.0, 0.0)
                reduction_tail = (reduction_chunk == reduction_chunks - 1) & (rows % BLOCK_K != 0)
                first_reduction = reduction_chunk == 0
                last_reduction = reduction_chunk == reduction_chunks - 1
                first = tile_in_expert == 0
                last = tile_in_expert == tiles_per_expert - 1
                repeated = tile_idx >= NUM_SM
                role_code = (
                    reduction_tail * 32
                    + first_reduction * 64
                    + last_reduction * 128
                    + first * 4
                    + last * 8
                    + repeated * 16
                )
                trace_offset = logical_idx * 6
                tl.store(trace_ptr + trace_offset, expert)
                tl.store(trace_ptr + trace_offset + 1, tile_m)
                tl.store(trace_ptr + trace_offset + 2, tile_n)
                tl.store(trace_ptr + trace_offset + 3, reduction_chunk)
                tl.store(trace_ptr + trace_offset + 4, role_code)
                tl.store(trace_ptr + trace_offset + 5, tl.program_id(0))
                reduction_chunk += 1

            output_offsets = (
                expert * HIDDEN * OUTPUT + offsets_m[:, None] * OUTPUT + offsets_n[None, :]
            )
            tl.store(
                grad_weight_ptr + output_offsets,
                accumulator,
                mask=(offsets_m[:, None] < HIDDEN) & (offsets_n[None, :] < OUTPUT),
            )
            tile_idx += NUM_SM

        row_start += rows
        logical_start += tiles_per_expert * reduction_chunks


def _linear_role(code: int) -> str:
    labels = []
    labels.append("m-tail" if code & _M_TAIL else "m-full")
    labels.append("n-tail" if code & _N_TAIL else "n-full")
    first = bool(code & _FIRST_IN_EXPERT)
    last = bool(code & _LAST_IN_EXPERT)
    if first and last:
        labels.append("expert-single-tile")
    elif first:
        labels.append("expert-first-tile")
    elif last:
        labels.append("expert-last-tile")
    else:
        labels.append("expert-interior-tile")
    labels.append("queue-repeat" if code & _QUEUE_REPEAT else "queue-first")
    return "/".join(labels)


def _weight_role(code: int) -> str:
    labels = []
    labels.append("reduction-tail" if code & _REDUCTION_TAIL else "reduction-full")
    first = bool(code & _FIRST_REDUCTION)
    last = bool(code & _LAST_REDUCTION)
    if first and last:
        labels.append("single-reduction")
    elif first:
        labels.append("first-reduction")
    elif last:
        labels.append("last-reduction")
    else:
        labels.append("interior-reduction")
    first_tile = bool(code & _FIRST_IN_EXPERT)
    last_tile = bool(code & _LAST_IN_EXPERT)
    if first_tile and last_tile:
        labels.append("expert-single-output-tile")
    elif first_tile:
        labels.append("expert-first-output-tile")
    elif last_tile:
        labels.append("expert-last-output-tile")
    else:
        labels.append("expert-interior-output-tile")
    labels.append("queue-repeat" if code & _QUEUE_REPEAT else "queue-first")
    return "/".join(labels)


def _linear_records(
    counts: tuple[int, ...],
    output: int,
    num_sms: int,
) -> list[dict[str, int | str]]:
    records = []
    global_tile = 0
    n_tiles = triton.cdiv(output, _BLOCK_N)
    for expert, rows in enumerate(counts):
        m_tiles = triton.cdiv(rows, _BLOCK_M)
        tiles = m_tiles * n_tiles
        for tile in range(tiles):
            tile_m, tile_n = divmod(tile, n_tiles)
            code = 0
            if tile_m == m_tiles - 1 and rows % _BLOCK_M:
                code |= _M_TAIL
            if tile_n == n_tiles - 1 and output % _BLOCK_N:
                code |= _N_TAIL
            if tile == 0:
                code |= _FIRST_IN_EXPERT
            if tile == tiles - 1:
                code |= _LAST_IN_EXPERT
            if global_tile >= num_sms:
                code |= _QUEUE_REPEAT
            records.append(
                {
                    "work_item": global_tile,
                    "expert": expert,
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "role": _linear_role(code),
                }
            )
            global_tile += 1
    return records


def _weight_records(
    counts: tuple[int, ...],
    hidden: int,
    output: int,
    num_sms: int,
) -> list[dict[str, int | str]]:
    records = []
    tiles_m = triton.cdiv(hidden, _BLOCK_M)
    tiles_n = triton.cdiv(output, _BLOCK_N)
    tiles_per_expert = tiles_m * tiles_n
    logical_start = 0
    for expert, rows in enumerate(counts):
        chunks = triton.cdiv(rows, _BLOCK_K)
        for tile in range(tiles_per_expert):
            tile_m, tile_n = divmod(tile, tiles_n)
            for chunk in range(chunks):
                code = 0
                if chunk == chunks - 1 and rows % _BLOCK_K:
                    code |= _REDUCTION_TAIL
                if chunk == 0:
                    code |= _FIRST_REDUCTION
                if chunk == chunks - 1:
                    code |= _LAST_REDUCTION
                if tile == 0:
                    code |= _FIRST_IN_EXPERT
                if tile == tiles_per_expert - 1:
                    code |= _LAST_IN_EXPERT
                if expert * tiles_per_expert + tile >= num_sms:
                    code |= _QUEUE_REPEAT
                records.append(
                    {
                        "work_item": logical_start + tile * chunks + chunk,
                        "expert": expert,
                        "tile_m": tile_m,
                        "tile_n": tile_n,
                        "reduction_chunk": chunk,
                        "role": _weight_role(code),
                    }
                )
        logical_start += tiles_per_expert * chunks
    return records


def _trace_records(
    trace: torch.Tensor,
    *,
    kind: str,
) -> list[dict[str, int | str]]:
    rows = trace.detach().cpu().tolist()
    decoder = _weight_role if kind == "weight" else _linear_role
    records = []
    for work_item, row in enumerate(rows):
        record: dict[str, int | str] = {
            "work_item": work_item,
            "expert": int(row[0]),
            "tile_m": int(row[1]),
            "tile_n": int(row[2]),
            "role": decoder(int(row[4])),
            "program_id": int(row[5]),
        }
        if kind == "weight":
            record["reduction_chunk"] = int(row[3])
        else:
            record["queue_iteration"] = int(row[3])
        records.append(record)
    return records


def _role_counts(records: list[dict[str, int | str]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["role"]) for record in records).items()))


def _pressure_class(work_items: tuple[int, ...], num_sms: int) -> str:
    def classify(value: int) -> str:
        if value < num_sms:
            return "underfilled"
        if value <= num_sms * 2:
            return "one-repeat"
        return "multi-repeat"

    return "|".join(classify(value) for value in work_items)


@dataclass(frozen=True)
class QualificationEvidence:
    kind: str
    declared_records: tuple[dict[str, int | str], ...]
    qualified_records: tuple[dict[str, int | str], ...]

    @property
    def declared_role_counts(self) -> dict[str, int]:
        return _role_counts(list(self.declared_records))

    @property
    def qualified_role_counts(self) -> dict[str, int]:
        return _role_counts(list(self.qualified_records))

    @property
    def derivation_digest(self) -> str:
        return _canonical_digest(
            {"source": "generated-metadata", "kind": self.kind, "records": self.declared_records}
        )

    @property
    def qualification_digest(self) -> str:
        return _canonical_digest(
            {"source": "device-trace", "kind": self.kind, "records": self.qualified_records}
        )

    def occurrences(self) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for record in self.qualified_records:
            grouped.setdefault(str(record["role"]), []).append(int(record["work_item"]))
        return {role: tuple(values) for role, values in sorted(grouped.items())}


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, counts, weight, backend):
        ctx.backend = backend
        ctx.save_for_backward(tokens, counts, weight)
        return backend.launch_forward(tokens, counts, weight)

    @staticmethod
    def backward(ctx, grad_output):
        tokens, counts, weight = ctx.saved_tensors
        grad_input, grad_weight = ctx.backend.launch_backward(
            tokens,
            counts,
            weight,
            grad_output.contiguous(),
        )
        return grad_input, None, grad_weight, None


class InstrumentedGroupedExperts(nn.Module):
    """Grouped linear layer with independently inspectable logical-work traces."""

    def __init__(
        self,
        *,
        num_experts: int = 4,
        hidden: int = 128,
        output: int = 128,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden = int(hidden)
        self.output = int(output)
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        weight = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.output,
                device=device,
                dtype=dtype,
            )
            / self.hidden**0.5
        )
        self.weight = nn.Parameter(weight)
        self.fault_kernel: str | None = None
        self.fault_work_item = -1
        self.last_traces: dict[str, torch.Tensor] = {}

    def forward(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        return _GroupedLinear.apply(tokens, counts, self.weight, self)

    def set_fault(self, kernel: str | None, work_item: int = -1) -> None:
        if kernel not in {None, "forward", "input-gradient", "weight-gradient"}:
            raise ValueError(f"unsupported fault kernel {kernel!r}")
        self.fault_kernel = kernel
        self.fault_work_item = int(work_item)

    def clear_fault(self) -> None:
        self.set_fault(None)

    def _validate_inputs(self, tokens: torch.Tensor, counts: torch.Tensor) -> tuple[int, ...]:
        values = tuple(int(value) for value in counts.detach().cpu().tolist())
        if len(values) != self.num_experts:
            raise ValueError("counts must contain one value per expert")
        if any(value < 0 for value in values):
            raise ValueError("counts must be non-negative")
        if sum(values) != tokens.shape[0]:
            raise ValueError("counts must sum to the packed token rows")
        if tokens.shape[1] != self.hidden:
            raise ValueError("token hidden dimension does not match the backend")
        return values

    def launch_forward(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        count_values = self._validate_inputs(tokens, counts)
        weight = self.weight if weight is None else weight
        output = torch.empty(
            tokens.shape[0],
            self.output,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        work_items = len(_linear_records(count_values, self.output, self.num_sms))
        trace = torch.empty(
            work_items,
            _TRACE_FIELDS,
            device=tokens.device,
            dtype=torch.int32,
        )
        _persistent_grouped_linear_kernel[(self.num_sms,)](
            tokens,
            weight,
            output,
            counts,
            trace,
            self.fault_work_item if self.fault_kernel == "forward" else -1,
            REDUCTION=self.hidden,
            OUTPUT=self.output,
            WEIGHT_EXPERT_STRIDE=self.hidden * self.output,
            WEIGHT_REDUCTION_STRIDE=self.output,
            WEIGHT_OUTPUT_STRIDE=1,
            NUM_EXPERTS=self.num_experts,
            NUM_SM=self.num_sms,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        self.last_traces["forward"] = trace
        return output

    def launch_backward(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        weight: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count_values = self._validate_inputs(tokens, counts)
        grad_input = torch.empty_like(tokens)
        input_work_items = len(_linear_records(count_values, self.hidden, self.num_sms))
        input_trace = torch.empty(
            input_work_items,
            _TRACE_FIELDS,
            device=tokens.device,
            dtype=torch.int32,
        )
        _persistent_grouped_linear_kernel[(self.num_sms,)](
            grad_output,
            weight,
            grad_input,
            counts,
            input_trace,
            self.fault_work_item if self.fault_kernel == "input-gradient" else -1,
            REDUCTION=self.output,
            OUTPUT=self.hidden,
            WEIGHT_EXPERT_STRIDE=self.hidden * self.output,
            WEIGHT_REDUCTION_STRIDE=1,
            WEIGHT_OUTPUT_STRIDE=self.output,
            NUM_EXPERTS=self.num_experts,
            NUM_SM=self.num_sms,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )

        grad_weight = torch.empty_like(weight)
        weight_work_items = len(
            _weight_records(count_values, self.hidden, self.output, self.num_sms)
        )
        weight_trace = torch.empty(
            weight_work_items,
            _TRACE_FIELDS,
            device=tokens.device,
            dtype=torch.int32,
        )
        _persistent_grouped_weight_grad_kernel[(self.num_sms,)](
            tokens,
            grad_output,
            grad_weight,
            counts,
            weight_trace,
            self.fault_work_item if self.fault_kernel == "weight-gradient" else -1,
            HIDDEN=self.hidden,
            OUTPUT=self.output,
            NUM_EXPERTS=self.num_experts,
            NUM_SM=self.num_sms,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        self.last_traces["input-gradient"] = input_trace
        self.last_traces["weight-gradient"] = weight_trace
        return grad_input, grad_weight

    def run_forward_backward(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.launch_forward(tokens, counts)
        grad_input, grad_weight = self.launch_backward(
            tokens,
            counts,
            self.weight,
            grad_output,
        )
        return output, grad_input, grad_weight

    def evidence(self, counts: tuple[int, ...]) -> tuple[QualificationEvidence, ...]:
        declarations = (
            _linear_records(counts, self.output, self.num_sms),
            _linear_records(counts, self.hidden, self.num_sms),
            _weight_records(counts, self.hidden, self.output, self.num_sms),
        )
        kinds = ("forward", "input-gradient", "weight-gradient")
        evidence = []
        for kind, declared in zip(kinds, declarations, strict=True):
            trace = self.last_traces.get(kind)
            if trace is None:
                raise RuntimeError(f"missing instrumented trace for {kind}")
            qualified = _trace_records(
                trace,
                kind="weight" if kind == "weight-gradient" else "linear",
            )
            if len(qualified) != len(declared):
                raise RuntimeError(
                    f"{kind} trace has {len(qualified)} records; expected {len(declared)}"
                )
            evidence.append(
                QualificationEvidence(
                    kind=kind,
                    declared_records=tuple(declared),
                    qualified_records=tuple(qualified),
                )
            )
        return tuple(evidence)

    def execution_hints(
        self,
        counts: tuple[int, ...],
        *,
        kernel_count: int,
        role_count_updates: dict[str, dict[str, int]] | None = None,
    ) -> ExecutionHints:
        evidence = self.evidence(counts)
        if kernel_count != len(evidence):
            raise RuntimeError(f"Kineto captured {kernel_count} kernels; expected {len(evidence)}")
        semantics = []
        work_items = []
        for item in evidence:
            declared_counts = item.declared_role_counts
            if role_count_updates and item.kind in role_count_updates:
                declared_counts = role_count_updates[item.kind]
            mapping = (
                "persistent-round-robin:cta->expert-output-tile;logical-work=reduction-chunk"
                if item.kind == "weight-gradient"
                else "persistent-round-robin:logical-tile->expert,m,n"
            )
            semantics.append(
                CTASemantics.create(
                    mapping_class=mapping,
                    role_counts=declared_counts,
                    qualification_role_counts=item.qualified_role_counts,
                    derivation_source="generated_metadata",
                    derivation_digest=item.derivation_digest,
                    qualification_source="instrumented",
                    qualification_digest=item.qualification_digest,
                )
            )
            work_items.append(len(item.declared_records))
        return ExecutionHints(
            algorithm_ids=(
                "triton-persistent-grouped-linear-forward",
                "triton-persistent-grouped-linear-input-gradient",
                "triton-persistent-grouped-linear-weight-gradient",
            ),
            tile_shapes=(
                f"{_BLOCK_M}x{_BLOCK_N}x{_BLOCK_K}",
                f"{_BLOCK_M}x{_BLOCK_N}x{_BLOCK_K}",
                f"{_BLOCK_M}x{_BLOCK_N}x{_BLOCK_K}",
            ),
            tail_path="masked-m-and-reduction-tails",
            workspace_bytes=0,
            pressure_class=_pressure_class(tuple(work_items), self.num_sms),
            overlap_class="none",
            persistent_work_items=tuple(work_items),
            cta_semantics=tuple(semantics),
            extra={
                "backend": "triton-tutorial-derived-grouped-gemm",
                "block": [_BLOCK_M, _BLOCK_N, _BLOCK_K],
                "num_experts": self.num_experts,
                "num_sms": self.num_sms,
            },
        )

    def role_occurrences(
        self,
        counts: tuple[int, ...],
    ) -> dict[str, dict[str, tuple[int, ...]]]:
        return {item.kind: item.occurrences() for item in self.evidence(counts)}
