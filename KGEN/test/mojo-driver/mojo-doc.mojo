# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# We don't allow input files that don't end in '.mojo' or '.🔥',
# including stdin.
# RUN: echo "" | not mojo-driver doc - 2>&1 | FileCheck %s --check-prefix CHECK-NOT-MOJO
# CHECK-NOT-MOJO: mojo-driver: cannot open '-', since it does not appear to be a Mojo file (it does not end in '.mojo' or '.🔥')

# When the input file cannot be opened, we print a nice error.
# RUN: not mojo-driver doc /does/not/exist.mojo 2>&1 | FileCheck %s --check-prefix CHECK-BAD-INPUT
# CHECK-BAD-INPUT: mojo-driver: cannot open input file '/does/not/exist.mojo': No such file or directory

# When the output file cannot be created or opened, we print a nice error.
# RUN: not mojo-driver doc %s -o no/such/directory.mojo 2>&1 | FileCheck %s --check-prefix CHECK-BAD-OUTPUT
# CHECK-BAD-OUTPUT: mojo-driver: cannot open output file 'no/such/directory.mojo': No such file or directory

# '-o /dev/null' works as expected.
# RUN: mojo-driver doc %s -o /dev/null

# Includes that point to files, and includes that point to nonexistent
# directories, are silently ignored.
# RUN: mojo-driver doc %s -I %s -I /does/not/exist
