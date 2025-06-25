# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import pathlib
import sys


def add_mojo_ipython_extension_to_sys_path() -> None:
    modular_dir = os.environ.get("MODULAR_PATH")
    assert modular_dir, "MODULAR_PATH env var must exist"

    dir = (
        pathlib.Path(modular_dir) / "KGEN" / "tools" / "mojo-ipython-extension"
    )

    assert dir.is_dir(), f"{dir} must be a directory"
    sys.path.append(dir.as_posix())
    return


add_mojo_ipython_extension_to_sys_path()
get_config().InteractiveShellApp.extensions = ["mojomagic"]
