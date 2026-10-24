// The leak checker reports nothing in production builds (SDLC-4780).
// UNSUPPORTED: production

// Check all the allocators:

// Default allocator will notice the crash.
// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: not mef-executor --allocator=leak-checker %t.mef 2>&1 | FileCheck %s

// Profiling allocator works like the default.
// RUN:  not mef-executor -allocator=profiler %t.mef

// CHECK: --- Running 'test_leak_one_int32'
grt.testcase @test_leak_one_int32 {
  "test.memory_leak_one_int32"() : () -> ()
  mef.output
}
// CHECK: Memory leak detected: 1 alive allocations, 4 alive bytes
