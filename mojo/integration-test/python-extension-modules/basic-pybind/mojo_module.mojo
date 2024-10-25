# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn arg_reg_trivial_borrowed(arg: Int):
    pass


fn incr_int(inout arg: Int):
    arg += 1


fn arg_memory_type(arg: String):
    pass


fn fill_string(inout arg: String):
    arg += "hello"
