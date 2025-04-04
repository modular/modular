# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn __wrap_and_execute_main[
    main_func: fn () -> None
](argc: Index, argv: Index) -> Index:
    return `0`


fn __wrap_and_execute_raising_main[
    main_func: fn () raises -> None
](argc: Index, argv: Index) -> Index:
    return `0`


fn __mojo_main_prototype(argc: Index, argv: Index) -> Index:
    return `0`
