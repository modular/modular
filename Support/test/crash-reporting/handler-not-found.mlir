// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp %crash-report-path-info %t/crash-report-path-info
// RUN: (env -i %t/crash-report-path-info -get crashpad-handler 2>&1; true) | FileCheck %s
// CHECK: could not determine crashpad handler path: {{.*}}
