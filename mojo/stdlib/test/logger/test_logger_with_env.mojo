# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D LOGGING_LEVEL=DEBUG %s | FileCheck %s

from logger import Logger


def main():
    var log = Logger()

    # CHECK: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK-NOT: INFO::: hello
    log.info("hello")

    # CHECK-NOT: CRITICAL::: hello
    log.critical("hello")
