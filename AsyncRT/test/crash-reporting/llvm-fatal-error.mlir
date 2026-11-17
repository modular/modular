// The LLVM diagnostic is what makes a report_fatal_error crash actionable, so
// check it reaches the crash database and not just stderr.
// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: printf '[crash_reporting]\nurl = http://invalid.\n' > %t/modular.cfg
// RUN: env MODULAR_HOME=%t crash-test-dummy -llvm-fatal-error > %t/output.txt 2>&1 || true
// RUN: FileCheck %s < %t/output.txt
// The message also lands in captured stack memory, so match it through the
// annotation name. The gap is the minidump string's length prefix.
// RUN: grep -a 'llvm_fatal_message.\{0,16\}Cannot select: synthetic-test' %t/crashdb/*/*.dmp
// CHECK: LLVM ERROR: Cannot select: synthetic-test
