#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file defines a simple utility script that checks if the current produced
# package differs from what has currently been released on the marketplace.
#
# ===----------------------------------------------------------------------=== #

import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

extension_dir = Path(__file__).parent.parent


def are_vsix_equal(lhs: Path, rhs: Path) -> bool:
    """Compare two .vsix files to see if they are structurally equivalent."""

    # A .vsix file is really just a .zip file, so we can extract and compare to
    # determine equivalence.
    with ZipFile(lhs, "r") as lhs_zip, ZipFile(rhs, "r") as rhs_zip:
        lhs_info = {info.filename: info for info in lhs_zip.infolist()}
        rhs_info = {info.filename: info for info in rhs_zip.infolist()}

        # Check that both .vsix files have the same internal files.
        if lhs_info.keys() != rhs_info.keys():
            return False

        # Ignore [Content_Types].xml, which can differ for a released vsix, but
        # isn't actually relevant for this check.
        filtered_names = [
            name for name in lhs_info.keys() if name != "[Content_Types].xml"
        ]

        # Compare the 32-bit CRCs of the archive files to see if they are the same.
        for name in filtered_names:
            if lhs_info[name].CRC != rhs_info[name].CRC:
                return False

        # The set of files and check sums are the same, compare the actual contents of
        # the files. Note that we could just compare the CRCs, but this is a more
        # thorough check (and still cheap enough to be fine).
        buffer_size = 1024
        for name in filtered_names:
            with (
                lhs_zip.open(lhs_info[name]) as lhs_file,
                rhs_zip.open(rhs_info[name]) as rhs_file,
            ):
                while True:
                    buffer1 = lhs_file.read(buffer_size)
                    buffer2 = rhs_file.read(buffer_size)
                    if buffer1 != buffer2:
                        return False
                    if not buffer1:
                        break

        return True


def get_current_version(extension_name: str) -> str:
    """Get the current version of the extension from the marketplace."""

    # Grab the currently released version from the marketplace.
    vsce_output: str = subprocess.run(
        ["vsce", "show", extension_name],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")

    # Grab the `Version: .*` line from the output.
    m = re.search(r"Version:\s*([0-9]+.[0-9]+.[0-9]+)\b", vsce_output)
    if not m:
        raise RuntimeError("could not find extension version in vsce output")
    return m.group(1)


def get_publisher_and_extension_name() -> tuple[str, str]:
    """Grab the extension name and publisher from the package.json."""
    with open(f"{extension_dir}/package.json") as f:
        package_json = f.read()

    # Grab the publisher.
    m = re.search(r'"publisher":\s*"([^"]+)"', package_json)
    if not m:
        raise RuntimeError("could not find publisher in package.json")
    publisher = m.group(1)

    # Grab the extension name.
    m = re.search(r'"name":\s*"([^"]+)"', package_json)
    if not m:
        raise RuntimeError("could not find name in package.json")
    extension_name = m.group(1)

    return publisher, extension_name


def main() -> None:
    publisher, extension_name = get_publisher_and_extension_name()
    full_extension_name = f"{publisher}.{extension_name}"

    # Grab the currently released version from the marketplace.
    version = get_current_version(full_extension_name)

    with TemporaryDirectory() as temp_dir:
        current_package = (
            Path(temp_dir) / f"current-{extension_name}-{version}.vsix"
        )
        released_package = (
            Path(temp_dir) / f"released-{extension_name}-{version}.vsix"
        )

        # Build the package with currently released version to better model the
        # released package.
        subprocess.run(
            [
                "vsce",
                "package",
                "-o",
                str(current_package),
                "--no-update-package-json",
                version,
            ],
            cwd=extension_dir,
            check=True,
        )

        # Grab the currently released package from the marketplace.
        subprocess.run(
            [
                "wget",
                "-q",
                f"https://{publisher}.gallery.vsassets.io/_apis/public/gallery/publisher/{publisher}/extension/{extension_name}/{version}/assetbyname/Microsoft.VisualStudio.Services.VSIXPackage",
                "-O",
                str(released_package),
            ],
            check=True,
        )

        # Compare the two packages.
        if not are_vsix_equal(current_package, released_package):
            sys.exit(1)


if __name__ == "__main__":
    main()
