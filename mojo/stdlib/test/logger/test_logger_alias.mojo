# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from logger import Level, Logger

alias log = Logger[Level.INFO]()


def main():
    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK: INFO::: hello
    log.info("hello")
