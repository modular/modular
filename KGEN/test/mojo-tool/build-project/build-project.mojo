# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that building our sample project results in 2 Mojo package artifacts:
# REQUIRES: DISABLED
# TODO(MOTO-503): Fix non-deterministic FileCheck failure to re-enable.
# RUN: mojo build-project %S/inputs/project 2>&1 | FileCheck %s
# CHECK: "id":0,{{.*}}"method":"build/initialize",{{.*}}"rootUri":"{{.*}}/inputs/project"
# CHECK: "id":0,{{.*}}"result":{{.*}}"displayName":"mojo-build-server"
# CHECK: "id":1,{{.*}}"method":"buildTarget/compile"
# CHECK: "id":1,{{.*}}"result":{{.*}}"statusCode":{{[1-3]}}
# RUN: test -f %S/inputs/project/.build/package_one.mojopkg
# RUN: test -f %S/inputs/project/.build/package_two.mojopkg
# RUN: rm -r %S/inputs/project/.build

# Test that `build-project` treats the current working directory as the
# workspace directory, by default. This directory contains no packages so the
# build is cancelled:
# RUN: not mojo build-project 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT-PATH
# CHECK-DEFAULT-PATH: "build/initialize",{{.*}}"rootUri":"{{.*}}/build-project"
# CHECK-DEFAULT-PATH: "result":{"statusCode":3}

# Test that an invalid workspace path is treated as an error:
# RUN: not mojo build-project path/does/not/exist 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-INVALID-PATH
# CHECK-INVALID-PATH: '{{.*}}/mojo-build-server{{.*}}' exited unsuccessfully
