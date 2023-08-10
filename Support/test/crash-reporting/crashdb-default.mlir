// RUN: rm -rf %t
// RUN: mkdir -p %t/home
// RUN: env -i MODULAR_HOME=%t/home %crash-report-path-info -get crashdb | FileCheck %s
// CHECK: {{.*}}home{{[\\/]}}crashdb
