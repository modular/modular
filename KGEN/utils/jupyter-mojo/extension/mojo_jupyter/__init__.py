# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._version import __version__


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "mojo_jupyter"}]
