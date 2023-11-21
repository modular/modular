# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait ATrait:
    """Some documentation."""

    fn print(owned self, x: StringRef):
        pass


struct Foo(ATrait):
    fn __init__(inout self):
        pass

    fn print(owned self, x: StringRef):
        pass
