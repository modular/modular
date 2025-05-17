# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from python import Python


fn arg_reg_trivial_borrowed(arg: Int):
    pass


fn incr_int(mut arg: Int):
    arg += 1


fn arg_memory_type(arg: String):
    pass


fn fill_string(mut arg: String):
    arg += "hello"
