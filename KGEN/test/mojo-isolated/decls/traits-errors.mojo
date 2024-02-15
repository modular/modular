# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s -verify-diagnostics


trait SomeTrait:
    # expected-error @+1 {{'self' argument must have type 'Self' in trait method declaration, but actually has type 'index'}}
    fn add(self: int):
        ...
