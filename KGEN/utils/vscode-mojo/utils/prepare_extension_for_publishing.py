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
import json
import shutil
from pathlib import Path

extension_dir = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="The version to use for the nightly build.",
    )
    parser.add_argument(
        "--kind",
        choices=["stable", "nightly"],
        required=True,
        help="The kind of release to prepare.",
    )

    args = parser.parse_args()

    stable = args.kind == "stable"
    nightly = not stable

    if stable:
        # Overwrite the icon with the stable icon.
        shutil.copy(
            extension_dir / "stable-icon.png",
            extension_dir / "icon.png",
        )

    # Update the package.json file to use the nightly version.
    package_json = extension_dir / "package.json"
    with open(package_json, "r") as f:
        package = json.load(f)

        # Update the various names to include "nightly".
        if stable:
            package["name"] = "vscode-mojo"
            package["displayName"] = "Mojo 🔥"
            package["description"] = "Mojo language support"
        if nightly:
            package["displayName"] = "Mojo 🔥 (nightly)"
            package["description"] = "Mojo language support (nightly)"
        package["version"] = args.version

        # Write the updated package.json file.
        with open(package_json, "w") as f:
            json.dump(package, f, ensure_ascii=False, indent=2)

    if nightly:
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
