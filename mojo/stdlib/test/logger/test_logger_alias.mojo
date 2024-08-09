# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from logger import Logger, Level

alias log = Logger[Level.INFO]()


def main():
    # CHECK: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK: INFO::: hello
    log.info("hello")

    # CHECK-NOT: CRITICAL::: hello
    log.critical("hello")
