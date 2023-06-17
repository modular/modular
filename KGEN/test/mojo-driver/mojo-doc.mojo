# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Reject unknown options.
# RUN: not mojo-driver doc -one --two 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo-driver{{.*}}: error: unrecognized argument '--two'

# We don't allow input files that don't end in '.mojo' or '.🔥',
# including stdin.
# RUN: echo "" | not mojo-driver doc - 2>&1 | FileCheck %s --check-prefix CHECK-NOT-MOJO
# CHECK-NOT-MOJO: mojo-driver{{.*}}: error: cannot open '-', since it does not appear to be a Mojo file (it does not end in '.mojo' or '.🔥')
#
# The user must provide an input.
# RUN: not mojo-driver doc 2>&1 | FileCheck %s --check-prefix CHECK-NO-INPUT
# CHECK-NO-INPUT: mojo-driver{{.*}}: error: no input file provided

# More than one input is not allowed.
# RUN: not mojo-driver doc %t.1.mojo %t.2.mojo 2>&1 | FileCheck %s --check-prefix CHECK-TOO-MANY-INPUT
# CHECK-TOO-MANY-INPUT: mojo-driver{{.*}}: error: too many input files, cannot process both '{{.*}}.1.mojo' and '{{.*}}.2.mojo'

# When the input file cannot be opened, we print a nice error.
# RUN: not mojo-driver doc /does/not/exist.mojo 2>&1 | FileCheck %s --check-prefix CHECK-BAD-INPUT
# CHECK-BAD-INPUT: mojo-driver{{.*}}: error: cannot open input file '/does/not/exist.mojo': No such file or directory

# When the output file cannot be created or opened, we print a nice error.
# RUN: not mojo-driver doc %s -o no/such/directory.mojo 2>&1 | FileCheck %s --check-prefix CHECK-BAD-OUTPUT
# CHECK-BAD-OUTPUT: mojo-driver{{.*}}: error: cannot open output file 'no/such/directory.mojo': No such file or directory

# '-o /dev/null' works as expected.
# RUN: mojo-driver doc %s -o /dev/null

# Includes that point to files, and includes that point to nonexistent
# directories, are silently ignored.
# RUN: mojo-driver doc %s -I %S/mojo-demangle.mojo -I /does/not/exist -o %t

# Validation itself is tested elsewhere; here we test only that the driver
# passes the `-validate` option through to the parser (this file contains
# validation warnings).
# RUN: mojo-driver doc -validate %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-VALIDATE
# CHECK-VALIDATE: mojo-doc.mojo:{{.*}}warning: unknown argument


fn f(x: Int):
    """This is an invalid doc string.

    Args:
      y: This argument doesn't appear in the argument list.
      z: Neither does this one.
    """
    pass
