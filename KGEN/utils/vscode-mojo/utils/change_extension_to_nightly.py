#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file defines a simple utility script that modifies the extension to
# prepare for packaging as a nightly release.
#
# ===----------------------------------------------------------------------=== #

import datetime
import json
from pathlib import Path

extension_dir = Path(__file__).parent.parent


def main():
    # Update the package.json file to use the nightly version.
    package_json = extension_dir / "package.json"
    with open(package_json, "r") as f:
        package = json.load(f)

        # Update the various names to include "nightly".
        package["name"] = "vscode-mojo-nightly"
        package["displayName"] = "Modular 🔥 (nightly)"
        package["description"] = "Mojo language support (nightly)"

        # The nightly version uses calver YYYY.MM.DDHH, update it.
        package["version"] = datetime.datetime.now().strftime("%Y.%-m.%-d%H")

        # Write the updated package.json file.
        with open(package_json, "w") as f:
            json.dump(package, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
