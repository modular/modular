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

    # CHECK: INFO::: hello
    log.info("hello")
