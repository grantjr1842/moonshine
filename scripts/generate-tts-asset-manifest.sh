#!/usr/bin/env bash
set -euo pipefail

root="${1:-core/moonshine-tts/data}"
output="${2:-core/moonshine-tts/data/manifest.json}"

python3 - "${root}" "${output}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
if not root.is_dir():
    raise SystemExit(f"tts manifest: root is not a directory: {root}")

assets = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"tts manifest: symlinked asset is not permitted: {path}")
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    # The root README documents the bundle but is not a native dependency key.
    if "/" not in relative:
        continue
    data = path.read_bytes()
    assets[relative] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }

if not assets:
    raise SystemExit(f"tts manifest: no publishable assets under {root}")

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "assets": assets,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

printf 'Wrote %s\n' "${output}"
