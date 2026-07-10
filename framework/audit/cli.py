from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.config import AuditRunConfig
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.scanner import AuditScanner
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.config import load_config
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.logging import configure_logging
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import suppress_warnings
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import PathPolicy


def add_audit_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("audit", help="Run a read-only raw data audit.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--raw-root", action="append", dest="raw_roots", help="Override/add raw root.")
    parser.add_argument("--output-root", help="Override output root.")
    parser.add_argument("--dataset", action="append", help="Dataset name or dataset_id to audit.")
    parser.add_argument("--run-id", help="Optional stable run id.")
    parser.add_argument("--max-files", type=int, help="Limit files scanned per dataset.")
    parser.add_argument("--max-dicom-headers", type=int, help="Limit DICOM headers read per dataset.")
    parser.add_argument("--workers", type=int, help="Reserved for parallel dataset scans.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    parser.set_defaults(func=run_from_args)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    suppress_warnings()
    configure_logging(args.log_level)
    settings, config_mapping = AuditRunConfig.from_file(
        args.config,
        raw_roots=args.raw_roots,
        output_root=args.output_root,
        max_files=args.max_files,
        max_dicom_headers=args.max_dicom_headers,
        workers=args.workers,
    )
    if args.output_root:
        config_mapping.setdefault("paths", {})["output_root"] = args.output_root
    if args.raw_roots:
        config_mapping.setdefault("paths", {})["raw_roots"] = args.raw_roots
    policy = PathPolicy.from_config(config_mapping, project_root=settings.project_root)
    scanner = AuditScanner(settings=settings, policy=policy)
    result = scanner.run(run_id=args.run_id, dataset_names=set(args.dataset or []))
    print(f"Audit complete: {result['output_dir']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run a read-only breast MRI data audit.")
    add_audit_parser(parser.add_subparsers(dest="command", required=True))
    prefix = [] if argv[:1] == ["audit"] else ["audit"]
    args = parser.parse_args([*prefix, *argv])
    return int(args.func(args))
