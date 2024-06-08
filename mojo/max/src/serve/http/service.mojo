# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Service definition for HTTP-based APIs."""

from python import PythonObject


trait PythonService:
    """A trait for Python-based HTTP servers."""

    fn handle(
        inout self, owned body: PythonObject, owned handler: PythonObject
    ) raises -> None:
        """Asynchronously process a single handler.

        Args:
            body: The decoded body (if POST).
            handler: The python handler object.
        """
        ...
