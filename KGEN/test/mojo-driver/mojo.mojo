# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the driver with an unknown subcommand also results in an error.
# RUN: not mojo-driver unknown 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo-driver{{.*}}: error: no such command 'unknown'

# Typos are diagnosed and similarly-typed commands are suggested.
# RUN: not mojo-driver domangle 2>&1 | FileCheck %s --check-prefix CHECK-TYPO
# CHECK-TYPO: mojo-driver{{.*}}: error: no such command 'domangle'. Did you mean 'demangle'?

# Invoking the driver with `--help` prints the driver's help text.
# RUN: mojo-driver --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: The Mojo{{.*}} command line interface
