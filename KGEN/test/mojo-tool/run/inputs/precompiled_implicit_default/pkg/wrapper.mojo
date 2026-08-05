# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# The implicit constructor is referenced only by the default value of
# `Container`'s `W` parameter, so a consumer that imports `Container` never
# materializes the constructor's declaration.


struct Wrapper(Copyable, Movable):
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value
