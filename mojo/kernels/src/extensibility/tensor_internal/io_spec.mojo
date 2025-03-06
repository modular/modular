# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct IO:
    var value: Int

    alias Output = IO(0)
    alias Input = IO(1)
    alias Unknown = IO(2)

    @always_inline("builtin")
    fn __init__(out self, value: Int):
        self.value = value


@value
@register_passable("trivial")
struct IOSpec[read: Bool, write: Bool]:
    """
    Parameter used to encode whether a particular tensor argument to a DPS kernel
    is an output, input, or mutable input.

    ```mojo
    Input == IOSpec[read=True, write=False]()
    Output == IOSpec[read=False, write=Write]()
    MutableInput == IOSpec[read=True, write=True]()
    ```
    """

    ...


alias IOUnknown = IOSpec[False, False]()

alias Input = IOSpec[True, False]()
alias Output = IOSpec[False, True]()
alias MutableInput = IOSpec[True, True]()
