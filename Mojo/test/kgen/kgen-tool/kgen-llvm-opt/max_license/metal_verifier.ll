; RUN: not kgen-llvm-opt -passes=kgen-metal-verifier %s -S 2>&1 | FileCheck %s

target triple = "air64-apple-macosx"

; A kernel using f64 must be rejected with a clear error.
; CHECK-DAG: argument 'x' has unsupported type 'double'
; CHECK-DAG: Function 'kernel_f64_arg' has Metal-unsupported instructions

; CHECK-DAG: function's return type 'double' is not supported
; CHECK-DAG: Function 'kernel_f64_ret' has Metal-unsupported instructions

; CHECK-DAG: instruction '{{.*}} fadd double{{.*}}' returns unsupported type 'double'
; CHECK-DAG: Function 'kernel_f64_inst' has Metal-unsupported instructions

; CHECK-DAG: argument 'arg0' has unsupported type 'double'
; CHECK-DAG: Function 'kernel_struct_f64_arg' has Metal-unsupported instructions

; CHECK-DAG: instruction '{{.*}} extractvalue { { double }, {} }{{.*}}' returns unsupported type 'double'
; CHECK-DAG: Function 'kernel_struct_f64' has Metal-unsupported instructions

; CHECK-DAG: argument 'arg' has unsupported type 'double'
; CHECK-DAG: Function 'kernel_vector_f64_arg' has Metal-unsupported instructions

; CHECK-DAG: instruction '{{.*}} call <2 x double> @llvm.vector.reverse.v2f64{{.*}}' returns unsupported type 'double'
; CHECK-DAG: Function 'kernel_vector_f64' has Metal-unsupported instructions

; CHECK-DAG: argument 'arg0' has unsupported type 'double'
; CHECK-DAG: Function 'kernel_array_f64_arg' has Metal-unsupported instructions

; CHECK-DAG: instruction '{{.*}} load double{{.*}}' returns unsupported type 'double'
; CHECK-DAG: Function 'kernel_array_f64' has Metal-unsupported instructions

; CHECK-DAG: function's return type 'double' is not supported
; CHECK-DAG: Function 'llvm.vector.reverse.v2f64' has Metal-unsupported instructions

; CHECK: LLVM ERROR: Failed to verify LLVM IR for Metal

define void @kernel_f64_arg(double %x) #0 {
  %r = fadd double %x, 1.0
  ret void
}

define double @kernel_f64_ret() #0 {
  ret double 1.0
}

define void @kernel_f64_inst() #0 {
  %r = fadd double 1.0, 1.3
  ret void
}

define void @kernel_struct_f64_arg({double} %arg0) #0 {
  ret void
}

define void @kernel_struct_f64() #0 {
  %ext = extractvalue { { double }, {} } zeroinitializer, 0, 0
  ret void
}

define void @kernel_vector_f64_arg(<2 x double> %arg) #0 {
  ret void
}

define void @kernel_vector_f64() #0 {
  %rev = call <2 x double> @llvm.vector.reverse.v2f64(<2 x double> zeroinitializer)
  ret void
}

define void @kernel_array_f64_arg([2 x double] %arg0) #0 {
  ret void
}

define void @kernel_array_f64() #0 {
  %array = alloca [8 x i8]
  %load = load double, ptr %array
  ret void
}

attributes #0 = { "metal.kernel"="true" }
