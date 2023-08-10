// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp %crash-report-path-info %t/crash-report-path-info
// RUN: cp %modular-crashpad-handler %t/nonstandard-handler
// RUN: printf '[crash_reporting]\nhandler_path = %t/nonstandard-handler\n' > %t/modular.cfg
// RUN: env MODULAR_HOME=%t %t/crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}nonstandard-handler
