# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not --crash %mojo %s | FileCheck %s


from logger import Level, Logger


def main():
    var log = Logger[Level.CRITICAL]()

    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK-NOT: DEBUG::: hello world
    log.info("hello", "world")

    # CHECK: CRITICAL::: hello
    log.critical("hello")
