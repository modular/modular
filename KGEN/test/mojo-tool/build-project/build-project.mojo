# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build-project 2>&1 | FileCheck %s
# CHECK: client build/initialize/reply: displayName='mojo-build-server'
