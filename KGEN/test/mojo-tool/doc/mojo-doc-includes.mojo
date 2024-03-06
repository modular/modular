# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Includes that point to files, and includes that point to nonexistent
# directories, are silently ignored.
# RUN: mojo doc %s -I %S/mojo-demangle.mojo -I /does/not/exist -o %t
