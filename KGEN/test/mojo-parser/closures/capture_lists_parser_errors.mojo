# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s


def make_closure(x: Int):
    # CHECK: error: expected a capture convention list
    def my_closure(y: Int) unified -> Int:
        return x + y


# // -----

# COM: Verify Capture Rules Are Enforced

def immByMut(byRefMut: String):
    # CHECK: error: Cannot capture byRefMut by mut because it could be immutable
    def myclosure() unified {mut byRefMut}:
        pass


struct DoNotMoveMe:
    pass

def notMovable(var byMove: DoNotMoveMe):
    # CHECK: error: Cannot capture byMove by move because the type is not movable
    def myclosure() unified {var byMove^}:
        pass

struct MoveMe:
    pass

def immByMov(var byMove: MoveMe):
    # CHECK: error: Cannot capture byMove by move because the type is not movable
    def myclosure() unified {var byMove^}:
        pass

def paramNotAllowed[X: Int, Y: Int]():
    # CHECK: error: value X is a parameter and does not need a capture convention
    # CHECK: error: value Y is a parameter and does not need a capture convention
    def myclosure() unified {var X, read Y}:
        pass


def doesNotExist():
    # CHECK: error: reference to an unknown value: What
    def myclosure() unified {var What}:
        pass


# // -----


def mutateMe(mut str: String):
    pass


def illegal(mut byRefMut: String):
    def myclosure() unified {read byRefMut}:
        # CHECK: invalid call to 'mutateMe': value passed to mutable argument 'str' must be mutable
        mutateMe(byRefMut)


# // -----


def toy(rogue: String):
    # CHECK: error: Could not infer capture convention of the captured value rogue
    def myclosure() unified {} -> String:
        return rogue
