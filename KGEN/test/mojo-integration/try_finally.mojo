# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s --debug-level full 2>&1 | FileCheck %s

from collections.string import StaticString


def try_it(c0: Bool, c1: Bool) -> StaticString:
    try:
        try:
            print("try")
            return "dead code"
        finally:
            print("finally")
            if c0:
                return "true!"
            return "false!"
    finally:
        print("finally again!")
        if c1:
            return "interrupt!"


struct MyCtxtMgr:
    var handle: Bool

    @implicit
    fn __init__(out self, handle: Bool = True):
        self.handle = handle

    fn __enter__(self):
        pass

    fn __exit__(self):
        print("exit!")

    fn __exit__(self, err: Error) -> Bool:
        print("exit error!")
        return self.handle


fn with_no_throw() -> Int:
    with MyCtxtMgr():
        return 1


def with_it() -> Int:
    # CHECK-NOT: warning: 'except' logic is unreachable, try doesn't raise an exception
    with MyCtxtMgr():
        return 2


def with_it_err(handle: Bool) -> Int:
    with MyCtxtMgr(handle):
        raise Error()
    return 3


@fieldwise_init
struct MemoryType(Copyable, Movable):
    fn __del__(owned self):
        print("delete")


fn chris_origin_example(a: Bool, b: Bool):
    print("start")
    var x: MemoryType
    try:
        try:
            if a:
                x = MemoryType()
                raise Error()
        finally:
            if b:
                print("early")
                return
    except:
        _ = x^  # Keep alive for the test
    print("normal")


def main():
    # CHECK-LABEL: == try-finally
    print("== try-finally")
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: true!
    print(try_it(True, False))
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: false!
    print(try_it(False, False))
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: interrupt!
    print(try_it(True, True))
    # CHECK-NEXT: try
    # CHECK-NEXT: finally
    # CHECK-NEXT: finally again!
    # CHECK-NEXT: interrupt!
    print(try_it(False, True))

    # CHECK-NEXT: exit!
    # CHECK-NEXT: 1
    print(with_no_throw())
    # CHECK-NEXT: exit!
    # CHECK-NEXT: 2
    print(with_it())
    # CHECK-NEXT: exit error!
    # CHECK-NEXT: 3
    print(with_it_err(True))
    try:
        # CHECK-NEXT: exit error!
        print(with_it_err(False))
    except:
        # CHECK-NEXT: an error was raised
        print("an error was raised")

    # CHECK-NEXT: start
    # CHECK-NEXT: delete
    # CHECK-NEXT: normal
    chris_origin_example(True, False)
    # CHECK-NEXT: start
    # CHECK-NEXT: early
    chris_origin_example(False, True)
    # CHECK-NEXT: start
    # CHECK-NEXT: delete
    # CHECK-NEXT: early
    chris_origin_example(True, True)
