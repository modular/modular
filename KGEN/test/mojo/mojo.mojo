# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the driver with an unknown subcommand results in an error.
# RUN: not mojo unknown 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo{{.*}}: error: no such command 'unknown'

# Typos are diagnosed and similarly-typed commands are suggested.
# RUN: not mojo domangle 2>&1 | FileCheck %s --check-prefix CHECK-TYPO
# CHECK-TYPO: mojo{{.*}}: error: no such command 'domangle'. Did you mean 'demangle'?

# Invoking the driver with `--help` prints the driver's help text.
# RUN: mojo --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: The Mojo{{.*}} command line interface
