# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from dataclasses import dataclass
from pathlib import Path

from modular.utils.subprocess import run_shell_command


@dataclass(repr=True)
class ImageID:
    # Type of image
    basename: str = "modular-benchmarking-images"
    # Image revision (full git sha or 'latest')
    sha: str = "latest"
    # Image architecture: intel|amd|graviton
    arch: str = "intel"


def get_modular_ecr_url(region: str = "us-east-1") -> str:
    return f"466483404629.dkr.ecr.{region}.amazonaws.com"


def get_image_url(img: ImageID, region: str = "us-east-1") -> str:
    base_url = get_modular_ecr_url(region)
    return f"{base_url}/{img.basename}:{img.sha}-{img.arch}"


def login(region: str = "us-east-1") -> None:
    base_url = get_modular_ecr_url(region)
    login = run_shell_command(
        f"aws ecr get-login-password --region {region}".split(),
        capture_output=True,
    )
    run_shell_command(
        f"docker login --username AWS --password-stdin {base_url}".split(),
        input=login.stdout,
    )


def prune(force: bool = False):
    command = "docker system prune --all"
    command += " --force" if force else ""
    return run_shell_command(command.split(" "), capture_output=True)


def pull(img: ImageID) -> None:
    url = get_image_url(img)
    run_shell_command(["docker", "pull", url])


def run(img: ImageID, exe: Path, args: str = ""):
    """
    Execute the given script inside the given benchmarking container.
    The parent dir of <exe> is mounted within the container.

    Args:
        exe (Path): Path to the bash script to be executed.
        args (str): Arguments to pass to bash script.
        sha (str):  Git SHA of desired image.
        arch (str): HW architecture of desired image (intel|amd|graviton).

    Returns:
        Captured output from command (undecoded).
    """
    url = get_image_url(img)
    basename = os.path.basename(exe)
    parentdir = os.path.dirname(exe)

    mount_args = f"-w /workspace -v {parentdir}:/workspace"
    output = run_shell_command(
        f"docker run {mount_args} {url} bash ./{basename} {args}".split(),
        check=False,
    )

    if output.returncode != 0:
        raise RuntimeError("Failed to execute shell command")

    return output
