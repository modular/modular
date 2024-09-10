# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s


from testing import assert_true

from utils.lock import SpinWaiter


def test_spin_waiter():
    var waiter = SpinWaiter()
    alias RUNS = 1000
    for _ in range(RUNS):
        waiter.wait()
    assert_true(True)


def main():
    test_spin_waiter()
