# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._version import __version__


def _jupyter_labextension_paths():  # noqa: ANN202
    return [{"src": "labextension", "dest": "mojo_jupyter"}]
