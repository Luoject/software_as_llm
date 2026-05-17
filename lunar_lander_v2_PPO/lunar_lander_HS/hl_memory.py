"""HL memory: trials log, compressed summaries, diff snapshots."""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "ts": datetime.now().isoformat(timespec="seconds")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def compress_trials(
    trials_path: Path,
    summary_path: Path,
    max_lines: int = 200,
) -> int:
    """Keep last max_lines in trials; write rolling stats to summary CSV-like jsonl."""
    if not trials_path.exists():
        return 0
    lines = trials_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= max_lines:
        return len(lines)

    archive = trials_path.with_suffix(trials_path.suffix + ".archive.gz")
    with gzip.open(archive, "at", encoding="utf-8") as gz:
        for line in lines[:-max_lines]:
            gz.write(line + "\n")

    kept = lines[-max_lines:]
    trials_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    scores = []
    for line in kept:
        try:
            scores.append(float(json.loads(line).get("best_mean_return", 0)))
        except (json.JSONDecodeError, TypeError):
            pass
    summary = {
        "compressed_at": datetime.now().isoformat(timespec="seconds"),
        "archived_lines": len(lines) - max_lines,
        "kept_lines": len(kept),
        "window_best_mean": max(scores) if scores else None,
        "window_last_mean": scores[-1] if scores else None,
    }
    append_jsonl(summary_path, summary)
    return len(kept)


def snapshot_policy(policy_path: Path, history_dir: Path, tag: str) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    dest = history_dir / f"policy_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(policy_path, dest)
    return dest
