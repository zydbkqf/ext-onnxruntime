#!/usr/bin/env bash
# Fails when package.xml and the tree have drifted apart.
#
# `pecl package` only ships what package.xml lists, so a file that is tracked but
# unlisted silently disappears from the release tarball: a new PHPT never runs
# for PECL users, a new docs page turns README links into 404s. A file listed
# but absent also breaks `pecl package`.
#
# Usage: bash tests/check-package-contents.sh   (run from the repo root)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$ROOT/package.xml" <<'PY'
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

package = sys.argv[1]
ns = {"p": "http://pear.php.net/dtd/package-2.0"}
root = ET.parse(package).getroot()
listed = {f.get("name") for f in root.iter("{http://pear.php.net/dtd/package-2.0}file")}
if not listed:
    listed = {f.get("name") for f in root.findall(".//file", ns)}

# Areas the release must cover. .github/, third_party/ and run-tests.php are
# developer-only and deliberately not packaged.
areas = ("src/", "tests/", "docs/", "examples/", "benchmarks/")
roots = ("config.m4", "config.w32", "README.md", "composer.json", "LICENSE", "NOTICE")

# Tracked plus not-yet-committed (but not ignored) files, so the check also
# catches drift before the next commit.
tracked = subprocess.run(
    ["git", "ls-files", "-co", "--exclude-standard"],
    capture_output=True, text=True, check=True,
).stdout.splitlines()

expected = {p for p in tracked if p.startswith(areas) or p in roots}

missing = sorted(expected - listed)
extra = sorted(f for f in listed - expected if not os.path.exists(f))

status = 0
for path in missing:
    print(f"check-package-contents: tracked but not in package.xml: {path}")
    status = 1
for path in extra:
    print(f"check-package-contents: listed in package.xml but missing on disk: {path}")
    status = 1

if status == 0:
    print(f"check-package-contents: {len(listed)} files listed, tree agrees")
sys.exit(status)
PY
