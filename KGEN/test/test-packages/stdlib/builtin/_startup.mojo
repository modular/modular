# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn __wrap_and_execute_main[
    main_func: fn () -> None
](argc: int, argv: int) -> int:
    return `0`


fn __wrap_and_execute_raising_main[
    main_func: fn () raises -> None
](argc: int, argv: int) -> int:
    return `0`


fn __wrap_and_execute_object_raising_main[
    main_func: fn () raises -> object
](argc: int, argv: int) -> int:
    return `0`


fn __mojo_main_prototype(argc: int, argv: int) -> int:
    return `0`
