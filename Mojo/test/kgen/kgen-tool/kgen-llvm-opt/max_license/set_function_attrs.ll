; Guards that the `kgen-set-function-attrs` pass (provided by NVPTXBackend) is
; registered for `-passes=` and applies the f32 denormal mode it is given.
; RUN: kgen-llvm-opt \
; RUN:   -passes='kgen-set-function-attrs<denormal-fp-math-f32=preserve-sign>' \
; RUN:   %s -S | FileCheck %s
; RUN: kgen-llvm-opt -passes=kgen-set-function-attrs %s -S \
; RUN:   | FileCheck %s --check-prefix=NOMODE
; The parameter must name the setting.
; RUN: not kgen-llvm-opt -passes='kgen-set-function-attrs<preserve-sign>' %s \
; RUN:   -S 2>&1 | FileCheck %s --check-prefix=UNNAMED

target triple = "nvptx64-nvidia-cuda"

; CHECK: define void @f() #[[ATTR:[0-9]+]]
; NOMODE: define void @f() {
; UNNAMED: unknown pass name 'kgen-set-function-attrs<preserve-sign>'
define void @f() {
  ret void
}

; CHECK: attributes #[[ATTR]] = { denormal_fpenv(float: preservesign) }
; NOMODE-NOT: denormal_fpenv
