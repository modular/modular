// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: printf '[crash_reporting]\nurl = http://invalid.\n' > %t/modular.cfg
// RUN: env MODULAR_HOME=%t crash-test-dummy || true
// RUN: (cd %t && find .) | FileCheck %s
// CHECK: ./crashdb/{{pending|completed}}/{{.*}}.dmp
