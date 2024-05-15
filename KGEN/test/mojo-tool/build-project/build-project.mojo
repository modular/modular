# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build-project 2>&1 | FileCheck %s
# CHECK: client reply:build/initialize(0): displayName='mojo-build-server'
