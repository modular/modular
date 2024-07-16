// RUN: build-info --query=modular-version | FileCheck %s --check-prefix=CHECK-MODULAR-VERSION
// RUN: build-info --query=git-revision | FileCheck %s --check-prefix=CHECK-GIT-REVISION
// RUN: build-info --query=build-type | FileCheck %s --check-prefix=CHECK-BUILD_TYPE
// RUN: build-info --query=kernels-build-type | FileCheck %s --check-prefix=CHECK-KERNELS-BUILD-TYPE
// RUN: build-info --query=asyncrt-max-profiling-level | FileCheck %s --check-prefix=CHECK-AsyncRT-MAX-PROFILING-LEVEL
// RUN: build-info --query=preferred-memory-alignment | FileCheck %s --check-prefix=CHECK-PREFERRED-MEM-ALIGNMENT
// RUN: build-info --query=llvm-targets | FileCheck %s --check-prefix=CHECK-LLVM-TARGETS

// CHECK-MODULAR-VERSION: {{.*}}
// CHECK-GIT-REVISION: {{.*}}
// CHECK-BUILD_TYPE: {{.*}}
// CHECK-KERNELS-BUILD-TYPE: {{.*}}
// CHECK-AsyncRT-MAX-PROFILING-LEVEL: {{[0-9]+}}
// CHECK-PREFERRED-MEM-ALIGNMENT: {{[0-9]+}}
// CHECK-LLVM-TARGETS: {{.*}}
