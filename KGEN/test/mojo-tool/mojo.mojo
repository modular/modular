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

# Invoking the driver with `--help` prints the driver's help text, which
# includes its subcommands.
# RUN: mojo --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: mojo

# Invoking the driver with `--version` prints the version, for example
# '0.4.0-release (eb70c661)':
# RUN: mojo --version | FileCheck %s --check-prefix CHECK-VERSION
# CHECK-VERSION: mojo {{[0-9]+}}.{{[0-9]+}}.{{[0-9]+}}{{.*}}({{[a-f0-9]+}})
