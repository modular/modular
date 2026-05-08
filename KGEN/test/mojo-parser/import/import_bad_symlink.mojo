# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t
# RUN: ln -s does-not-exist %t/non_existent_package.mojo
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t %s

# expected-error-re @+1 {{unable to resolve imported module '{{.*}}non_existent_package.mojo'}}
import non_existent_package

def main():
    pass
