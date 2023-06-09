# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the driver without specifying any subcommands results in an error.
# RUN: not mojo-driver | FileCheck %s
# CHECK: USAGE: mojo-driver [options]

# Invoking the driver with an unknown subcommand also results in an error.
# RUN: not mojo-driver unknown 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo-driver: Unknown command line argument 'unknown'.  Try: 'mojo-driver --help'
