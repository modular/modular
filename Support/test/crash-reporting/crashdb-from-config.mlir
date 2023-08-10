// RUN: rm -rf %t
// RUN: mkdir -p %t/home
// RUN: printf '[crash_reporting]\ndatabase_path = %t/nonstandard-crashdb\n' > %t/modular.cfg
// RUN: env -i MODULAR_HOME=%t %crash-report-path-info -get crashdb | FileCheck %s
// CHECK: {{.*}}nonstandard-crashdb
