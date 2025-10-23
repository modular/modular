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

# COM: Verify Capture Rules Are Enforced

fn immByMut(byRefMut: String):
    # CHECK: error: Cannot capture byRefMut by mut because the value is immutable
    fn myclosure() unified {mut byRefMut}:
        pass


struct DoNotMoveMe:
    pass

fn notMovable(var byMove: DoNotMoveMe):
    # CHECK: error: Cannot capture byMove by move because the type is not movable
    fn myclosure() unified {var byMove^}:
        pass

struct MoveMe:
    pass

fn immByMov(var byMove: MoveMe):
    # CHECK: error: Cannot capture byMove by move because the type is not movable
    fn myclosure() unified {var byMove^}:
        pass

fn paramNotAllowed[X: Int, Y: Int]():
    # CHECK: error: value X is a parameter and does not need a capture convention
    # CHECK: error: value Y is a parameter and does not need a capture convention
    fn myclosure() unified {var X, read Y}:
        pass


fn doesNotExist():
    # CHECK: error: reference to an unknown value: What
    fn myclosure() unified {var What}:
        pass


# // -----


fn mutateMe(mut str: String):
    pass


fn illegal(mut byRefMut: String):
    fn myclosure() unified {read byRefMut}:
        # CHECK: error: invalid call to 'mutateMe': argument #0 must be mutable in order to pass to a mutating argument
        mutateMe(byRefMut)


# // -----


fn toy(rogue: String):
    # CHECK: error: Could not infer capture convention of the captured value rogue
    fn myclosure() unified {} -> String:
        return rogue

# // -----

fn wrongConvention(thing: String):
    # CHECK: error: Unrecognized capture convention
    fn incorrectCaptureConvention() unified {ref thing}:
        _ = thing
