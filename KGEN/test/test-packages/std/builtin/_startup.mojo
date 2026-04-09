# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def __wrap_and_execute_main[
    main_func: def() thin -> None
](argc: Int, argv: Int) -> Int:
    return 0


def __wrap_and_execute_raising_main[
    main_func: def() thin raises -> None
](argc: Int, argv: Int) -> Int:
    return 0


def __mojo_main_prototype(argc: Int, argv: Int) -> Int:
    return 0
