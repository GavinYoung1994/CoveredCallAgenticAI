"""Per-run artifact logging.

Every screener run persists, regardless of whether Discord is configured:
  * ``<run_id>.md``   — the full human-readable recommendation summary, and
  * ``<run_id>.json`` — the structured proposals with status PENDING_APPROVAL.

The full summary is also echoed to the server log. The JSON file is the record
the human-feedback CLI reads to capture approve/deny decisions before anything
is written to the SQL decision log (design §3).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger("runlog")


def format_rejections(rejected: List[Dict[str, Any]]) -> str:
    """Build a human-readable rejection breakdown for the run .md.

    Grouped by stage (SCOUT/QUANT/NEWS/RISK_MANAGER), with a reason tally per
    stage (so you can see *why* the universe collapsed at a glance) followed by
    the full per-symbol detail.
    """
    if not rejected:
        return "\n\n## Rejections\n_None — every candidate passed every guardrail._"

    by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rejected:
        by_stage[r.get("stage", "?")].append(r)

    out = [f"\n\n## Rejections ({len(rejected)})"]
    for stage in sorted(by_stage):
        items = by_stage[stage]
        out.append(f"\n### {stage} — {len(items)}")
        # Coarse reason tally (text before the first ':' or '(').
        tally = Counter(
            (it.get("reason", "").split(":")[0].split("(")[0].strip() or "unspecified")
            for it in items
        )
        out.append("**Reason tally:**")
        for cat, n in tally.most_common():
            out.append(f"- {cat}: {n}")
        out.append("\n**Detail:**")
        for it in items:
            line = f"- {it.get('symbol')}: {it.get('reason')}"
            srcs = it.get("sources") or []
            if srcs:  # NEWS rejections carry the headlines reviewed
                for s in srcs[:3]:
                    line += f"\n    - [{s.get('publisher')}] {s.get('title')} {s.get('url') or ''}".rstrip()
            out.append(line)
    return "\n".join(out)


def save_run(
    run_id: str,
    summary: str,
    recommendations: List[Dict[str, Any]],
    *,
    run_timestamp: str = "",
    rejected: Optional[List[Dict[str, Any]]] = None,
    workflow: str = "entry_screener",
    runs_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Write the run's summary + proposals to disk and echo to the log.

    Includes a full rejection breakdown (by stage + reason tally) in the .md and
    the raw rejected list in the .json, so you can always see WHY candidates were
    dropped. Never raises on a logging failure — a disk hiccup can't kill a run.
    """
    out_dir = Path(runs_dir or settings.runs_dir)
    paths: Dict[str, str] = {}
    rejected = rejected or []

    # The .md gets recommendations + the rejection breakdown appended.
    md_content = summary + format_rejections(rejected)

    # 1) Always echo to the server log (not truncated).
    logger.info("=== Run summary %s ===\n%s", run_id, md_content)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f"{run_id}.md"
        md_path.write_text(md_content, encoding="utf-8")
        paths["markdown"] = str(md_path)

        json_path = out_dir / f"{run_id}.json"
        payload = {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "workflow": workflow,
            "status": "PENDING_APPROVAL",
            "summary": summary,
            "recommendations": recommendations,
            "rejected": rejected,
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        paths["json"] = str(json_path)
        logger.info("Run artifacts saved: %s, %s", md_path, json_path)
    except OSError as exc:
        logger.error("Failed to persist run artifacts for %s: %s", run_id, exc)

    return paths


def load_run(run_id: str, runs_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load a saved run's proposals JSON (used by the feedback CLI). None if absent."""
    json_path = Path(runs_dir or settings.runs_dir) / f"{run_id}.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))
