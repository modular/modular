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

import argparse
import datetime
import json
import shutil
from pathlib import Path

extension_dir = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        help="The version to use for the nightly build.",
        # The nightly version uses calver. For example, a release dated
        # Feb 3 2024 1AM would be v2024.2.301.
        default=datetime.datetime.utcnow().strftime("%Y.%-m.%-d%H"),
    )
    args = parser.parse_args()

    # Overwrite the icon with the nightly icon.
    shutil.copy(
        extension_dir / "nightly-icon.png",
        extension_dir / "icon.png",
    )

    # Update the package.json file to use the nightly version.
    package_json = extension_dir / "package.json"
    with open(package_json, "r") as f:
        package = json.load(f)

        # Update the various names to include "nightly".
        package["name"] = "vscode-mojo-nightly"
        package["displayName"] = "Mojo 🔥 (nightly)"
        package["description"] = "Mojo language support (nightly)"
        package["version"] = args.version

        # Write the updated package.json file.
        with open(package_json, "w") as f:
            json.dump(package, f, ensure_ascii=False, indent=2)

    readme = extension_dir / "README.md"
    readme_prefix = """# Mojo for Visual Studio Code - Nightly

    > Attention: this is the nightly build of the vscode-mojo extension used for early feedback and testing.

    > Note: this extension requires that the stable vscode-mojo extension is not enabled on the editor.
    """
    with open(readme, "r") as f:
        text = readme_prefix + "".join(f.readlines()[1:])
        with open(readme, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
