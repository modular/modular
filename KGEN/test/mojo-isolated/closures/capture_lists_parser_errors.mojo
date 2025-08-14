# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s


fn make_closure(x: Int):
    # CHECK: error: expected a capture convention list
    fn my_closure(y: Int) unified -> Int:
        return x + y


# // -----

fn illegal(byRefMut: String):
    # CHECK: error: Cannot capture byRefMut by mut because the value is immutable
    fn myclosure() unified {mut byRefMut}:
        pass

# // -----

fn mutateMe(mut str:String):
    pass


fn illegal(mut byRefMut: String):
    fn myclosure() unified {read byRefMut}:
        # CHECK: error: invalid call to 'mutateMe': argument #0 must be mutable in order to pass to a mutating argument
        mutateMe(byRefMut)
