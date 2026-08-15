import pytest

from benchmarks.healthy_path import (
    _process_peak_rss_bytes,
    aggregate_step_latencies,
    percentile,
    replication_jump,
)
from benchmarks.run_healthy_path import _write_checksums, regression_percent, summarize_results


def test_percentile_interpolates_samples():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_step_latency_uses_slowest_rank_for_each_offset():
    results = [
        {"step_times_ms": [1.0, 20.0, 3.0]},
        {"step_times_ms": [2.0, 4.0, 30.0]},
    ]

    assert aggregate_step_latencies(results) == [2.0, 20.0, 30.0]


def test_gemini_replication_rejects_odd_multi_rank_topology():
    assert replication_jump(1) == 1
    assert replication_jump(2) == 1
    assert replication_jump(4) == 2
    with pytest.raises(ValueError, match="even world size"):
        replication_jump(3)


def test_process_peak_rss_reads_linux_high_water_mark(tmp_path):
    status = tmp_path / "123" / "status"
    status.parent.mkdir()
    status.write_text("Name:\tscout\nVmHWM:\t2048 kB\n")

    assert _process_peak_rss_bytes(123, proc_root=tmp_path) == 2 * 1024 * 1024


def test_regression_direction():
    assert regression_percent(100.0, 95.0, direction="higher") == 5.0
    assert regression_percent(100.0, 105.0, direction="lower") == 5.0


def test_summary_reports_threshold_violations():
    runs = {
        "baseline": {"commit_sha": "abc", "metrics": {"throughput": 100.0, "p95": 10.0}},
        "gemini": {"metrics": {"throughput": 96.0, "p95": 10.5}},
        "scout": {"metrics": {"throughput": 98.0, "p95": 10.2}},
        "combined": {"metrics": {"throughput": 90.0, "p95": 12.0}},
    }
    thresholds = {
        "metrics": {
            "throughput": {
                "direction": "higher",
                "maximum_regression_percent": {
                    "gemini": 5.0,
                    "scout": 3.0,
                    "combined": 7.5,
                },
            },
            "p95": {
                "direction": "lower",
                "maximum_regression_percent": {
                    "gemini": 10.0,
                    "scout": 7.5,
                    "combined": 15.0,
                },
            },
        }
    }

    summary = summarize_results(runs, thresholds)

    assert summary["status"] == "failed"
    assert {(item["mode"], item["metric"]) for item in summary["violations"]} == {
        ("combined", "throughput"),
        ("combined", "p95"),
    }


def test_checksums_cover_each_result_once(tmp_path):
    (tmp_path / "baseline.json").write_text("{}\n")
    (tmp_path / "summary.json").write_text("{}\n")

    _write_checksums(tmp_path)
    _write_checksums(tmp_path)

    lines = (tmp_path / "checksums.txt").read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("  baseline.json")
    assert lines[1].endswith("  summary.json")
