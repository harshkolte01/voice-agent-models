"""Generate a hashed API key and print the plaintext once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.keys import create_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an STT API key. The plaintext is shown once; only the hash is stored."
    )
    parser.add_argument("--name", required=True, help="Human-readable label, e.g. 'Developer A'")
    parser.add_argument("--id", dest="key_id", default=None, help="Optional stable id (default: slug of --name)")
    parser.add_argument(
        "--keys-file",
        default=None,
        help="Override keys file path (default: STT_KEYS_FILE / keys.json)",
    )
    args = parser.parse_args()

    settings = get_settings()
    keys_file = Path(args.keys_file or settings.stt_keys_file)

    try:
        plaintext, record = create_key(keys_file, name=args.name, key_id=args.key_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Save this key now; it will not be shown again.")
    print(f"id:          {record.id}")
    print(f"name:        {record.name}")
    print(f"prefix:      {record.prefix}")
    print(f"keys file:   {keys_file}")
    print(f"STT_API_KEY={plaintext}")


if __name__ == "__main__":
    main()
