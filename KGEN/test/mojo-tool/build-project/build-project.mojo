# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build-project 2>&1 | FileCheck %s
# CHECK: "id":0,{{.*}}"method":"build/initialize",{{.*}}"rootUri":"{{.*}}/build-project"
# CHECK: "id":0,{{.*}}"result":{{.*}}"displayName":"mojo-build-server"
# CHECK: "id":1,{{.*}}"method":"buildTarget/compile"
# CHECK: "id":1,{{.*}}"result":{{.*}}"statusCode":{{[1-3]}}

# RUN: mojo build-project %S 2>&1 | FileCheck %s --check-prefix=CHECK-PATH
# CHECK-PATH: "rootUri":"{{.*}}/build-project"

# RUN: not mojo build-project path/does/not/exist 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-INVALID-PATH
# CHECK-INVALID-PATH: '{{.*}}/mojo-build-server{{.*}}' exited unsuccessfully
