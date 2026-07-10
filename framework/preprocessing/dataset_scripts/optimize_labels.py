from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.label_optimization import optimize_label_tree  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize processed labels.json files by keeping core labels in front "
            "and moving secondary labels/features to sibling label_copy.json files."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Root directory or a single labels.json to process. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Only process this dataset id. Can be repeated.",
    )
    parser.add_argument(
        "--important-key",
        action="append",
        default=[],
        help="Keep an extra key in labels.json instead of moving it to label_copy.json.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Write changes. Omit for dry-run.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .label_optimize.bak files before overwriting existing JSON files.",
    )
    args = parser.parse_args(argv)

    roots = args.root or [
        str(PROJECT_ROOT / "outputs" / "local" / "multimodal_dataset"),
        str(PROJECT_ROOT / "outputs" / "local" / "unified_dataset"),
    ]
    summary = optimize_label_tree(
        roots,
        dry_run=not args.apply,
        backup=args.backup,
        dataset_ids=args.dataset_id,
        extra_important_keys=args.important_key,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
