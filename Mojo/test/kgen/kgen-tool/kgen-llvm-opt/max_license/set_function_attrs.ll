; Guards that the `kgen-set-function-attrs` pass (provided by NVPTXBackend) is
; registered for `-passes=` and applies the denormal-fp-math-f32 function
; attribute.
; RUN: kgen-llvm-opt -passes=kgen-set-function-attrs \
; RUN:   -denormal-fp-math-f32=preserve-sign %s -S | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

; CHECK: define void @f() #[[ATTR:[0-9]+]]
define void @f() {
  ret void
}

; CHECK: attributes #[[ATTR]] = { denormal_fpenv(float: preservesign) }
