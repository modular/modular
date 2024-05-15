# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build-project 2>&1 | FileCheck %s
# CHECK: build/initialize: {{.*}} rootUri='{{.*}}/build-project'
# CHECK: reply:build/initialize(0): displayName='mojo-build-server'

# RUN: mojo build-project %S 2>&1 | FileCheck %s --check-prefix=CHECK-PATH
# CHECK-PATH: build/initialize: {{.*}} rootUri='{{.*}}/build-project'

# RUN: not mojo build-project path/does/not/exist 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-INVALID-PATH
# CHECK-INVALID-PATH: '{{.*}}/mojo-build-server{{.*}}' exited unsuccessfully
