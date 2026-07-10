from __future__ import annotations

import argparse
from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.transfer.bundle import create_processed_data_bundle, verify_bundle
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import suppress_warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package and verify processed data uploads.")
    add_transfer_parser(parser.add_subparsers(dest="command", required=True))
    return parser


def add_transfer_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("transfer", help="Package and verify processed-data uploads.")
    transfer_subparsers = parser.add_subparsers(dest="transfer_command", required=True)

    package = transfer_subparsers.add_parser("package", help="Create a processed-data upload package.")
    package.add_argument("--config", default="configs/local_preprocess.yaml")
    package.add_argument("--manifest")
    package.add_argument("--processed-root")
    package.add_argument("--output-dir")
    package.add_argument("--bundle-name")
    package.add_argument("--zip", action="store_true", dest="make_zip")
    package.set_defaults(func=_package)

    verify = transfer_subparsers.add_parser("verify", help="Verify uploaded processed data on the server.")
    verify.add_argument("--bundle-dir", required=True)
    verify.add_argument("--processed-root", required=True)
    verify.add_argument("--output-json")
    verify.set_defaults(func=_verify)
    return parser


def _package(args: argparse.Namespace) -> int:
    suppress_warnings()
    result = create_processed_data_bundle(
        config_path=args.config,
        manifest_path=args.manifest,
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        bundle_name=args.bundle_name,
        make_zip=args.make_zip,
    )
    print(f"Bundle directory: {result.bundle_dir}")
    print(f"Files: {result.file_count}")
    print(f"Total bytes: {result.total_bytes}")
    if result.zip_path:
        print(f"Zip: {result.zip_path}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    suppress_warnings()
    result = verify_bundle(args.bundle_dir, args.processed_root)
    if args.output_json:
        atomic_write_json(Path(args.output_json), result)
    print(f"Checked: {result['checked']}")
    print(f"Missing: {len(result['missing'])}")
    print(f"Mismatched: {len(result['mismatched'])}")
    print(f"OK: {result['ok']}")
    return 0 if result["ok"] else 2


def main(argv: list[str] | None = None) -> int:
    suppress_warnings()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
