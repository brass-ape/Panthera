"""
manage_plugins.py
==================

Human-in-the-loop approval step for agent-proposed plugins (see
plugins.py's module docstring for the full flow). This script is the
ONLY way a proposed plugin ever reaches the trusted plugins/
directory that plugins.register_loaded_plugins() imports from --
nothing the agent does on its own can move a file there.

Usage:
    python manage_plugins.py list
    python manage_plugins.py show <name>
    python manage_plugins.py approve <name>
    python manage_plugins.py reject <name>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from config import CONFIG
from plugins import PLUGINS_DIR

PROPOSED_DIR = CONFIG.vault_dir / "plugins_proposed"

# Same charset the model's proposals are validated against (see
# parser.py's _PLUGIN_NAME_RE) -- kept here too so a mistyped name at
# the CLI fails fast instead of doing something surprising with path
# construction.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_valid_name(name: str) -> None:
    if not _NAME_RE.match(name):
        print(f"Invalid plugin name {name!r} (expected lowercase, e.g. 'roll_dice').", file=sys.stderr)
        sys.exit(1)


def _source_path(name: str) -> Path:
    return PROPOSED_DIR / f"{name}.py"


def _manifest_path(name: str) -> Path:
    return PROPOSED_DIR / f"{name}.manifest.json"


def cmd_list(_args: argparse.Namespace) -> None:
    proposed = sorted(PROPOSED_DIR.glob("*.manifest.json")) if PROPOSED_DIR.is_dir() else []
    approved = sorted(PLUGINS_DIR.glob("*.py")) if PLUGINS_DIR.is_dir() else []

    print("Pending proposals (vault/plugins_proposed/):")
    if not proposed:
        print("  (none)")
    for manifest_file in proposed:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            print(f"  {manifest['name']} -- {manifest['description']} (proposed {manifest['proposed_at']})")
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"  {manifest_file.stem}: [could not read manifest: {exc}]")

    print("\nApproved (active) plugins (plugins/):")
    active = [p for p in approved if not p.name.startswith("_")]
    if not active:
        print("  (none)")
    for path in active:
        print(f"  {path.stem}  ({path})")


def cmd_show(args: argparse.Namespace) -> None:
    _require_valid_name(args.name)
    source = _source_path(args.name)
    if not source.exists():
        print(f"No pending proposal named {args.name!r}.", file=sys.stderr)
        sys.exit(1)

    manifest_file = _manifest_path(args.name)
    if manifest_file.exists():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            print(f"Name: {data.get('name', args.name)}")
            print(f"Description: {data.get('description', '(none)')}")
            print(f"Proposed at: {data.get('proposed_at', '(unknown)')}")
        except (OSError, json.JSONDecodeError):
            pass
    print("-" * 60)
    print(source.read_text(encoding="utf-8"))


def cmd_approve(args: argparse.Namespace) -> None:
    _require_valid_name(args.name)
    source = _source_path(args.name)
    if not source.exists():
        print(f"No pending proposal named {args.name!r}.", file=sys.stderr)
        sys.exit(1)

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    destination = PLUGINS_DIR / f"{args.name}.py"
    if destination.exists() and not args.force:
        print(f"{destination} already exists. Re-run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    shutil.copyfile(source, destination)
    source.unlink()
    _manifest_path(args.name).unlink(missing_ok=True)

    print(f"Approved: copied to {destination}.")
    print("Restart the assistant (main.py / webapp.py) for it to take effect.")


def cmd_reject(args: argparse.Namespace) -> None:
    _require_valid_name(args.name)
    source = _source_path(args.name)
    if not source.exists():
        print(f"No pending proposal named {args.name!r}.", file=sys.stderr)
        sys.exit(1)
    source.unlink()
    _manifest_path(args.name).unlink(missing_ok=True)
    print(f"Rejected and removed proposal {args.name!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and approve agent-proposed plugins.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List pending proposals and approved plugins.").set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Print a proposed plugin's source code.")
    show_parser.add_argument("name")
    show_parser.set_defaults(func=cmd_show)

    approve_parser = subparsers.add_parser("approve", help="Move a proposal into plugins/ (trusted).")
    approve_parser.add_argument("name")
    approve_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing approved plugin of the same name."
    )
    approve_parser.set_defaults(func=cmd_approve)

    reject_parser = subparsers.add_parser("reject", help="Delete a proposal without approving it.")
    reject_parser.add_argument("name")
    reject_parser.set_defaults(func=cmd_reject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
