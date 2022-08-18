// RUN: kgen-opt -lower-hlkgen %s | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator(%arg0: si32) -> si32 {
// CHECK-NEXT:    kgen.return %arg0 : si32
// CHECK-NEXT:  }
hlkgen.generator @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

kgen.generator.interface @add<ty: dtype>(%arg0: !meta.scalar<ty>, %arg1: !meta.scalar<ty>)
-> !meta.scalar<ty>

// This implementation is fine.
// CHECK-LABEL: kgen.generator @add_f32<ty: dtype>(%arg0: !meta.scalar<ty>, %arg1: !meta.scalar<ty>) -> !meta.scalar<ty>
// CHECK-NEXT: constraints <eq(:dtype ty, f32), "f32 feels great">
// CHECK-NEXT: implements @add {
kgen.generator @add_f32<ty: dtype>(%arg0 : !meta.scalar<ty>, %arg1 : !meta.scalar<ty>) -> !meta.scalar<ty>
  constraints <eq(:dtype ty, f32), "f32 feels great"> implements @add {

  // CHECK: %0 = meta.cast_to_builtin %arg0 : !meta.scalar<ty> to f32
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<ty> to f32
  // CHECK: %1 = meta.cast_to_builtin %arg1 : !meta.scalar<ty> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<ty> to f32

  // CHECK: %2 = llvm.fadd %0, %1 : f32
  %2 = llvm.fadd %0, %1 : f32

  // CHECK: %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<ty>
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<ty>
  // CHECK: kgen.return %3
  kgen.return %3 : !meta.scalar<ty>
}

// This should be fine, but we're missing meta.scalar bind ops.
hlkgen.generator @add_64<ty: dtype>(%arg0 : !meta.scalar<f64>, %arg1 : !meta.scalar<f64>)
    -> !meta.scalar<ty> implements @add {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f64> to f64
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f64> to f64
  %2 = llvm.fadd %0, %1 : f64
  %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<ty>
  kgen.return %3 : !meta.scalar<ty>
}

// CHECK-LABEL: kgen.generator @add_64_thunk
// CHECK-NEXT:    constraints <eq(:dtype ty, f64), "argument #0 specifies 'ty' = f64">
// CHECK-NEXT:    implements @add {
// CHECK-NEXT:    %0 = meta.scalar.rebind %arg0
// CHECK-NEXT:    %1 = meta.scalar.rebind %arg1
// CHECK-NEXT:    %2 = kgen.call @add_64<ty: dtype = ty>(%0, %1)

//===----------------------------------------------------------------------===//
// Infer simple constraints.
//===----------------------------------------------------------------------===//

kgen.generator.interface @bufitf<size, ty: dtype -> xyz>(!meta.buffer<size, ty>) -> index

// This implementation infers that the ty argument must be f32.
hlkgen.generator @impl1<size, ty: dtype -> xyz>(%arg0: !meta.buffer<size, f32>) -> index
  implements @bufitf {
  %0 = meta.buffer.size %arg0 : !meta.buffer<size, f32>
  kgen.return<xyz = add(size, 2)> %0 : index
}

// This causes synthesization of a thunk for impl1 that adapts the calling convention.

// CHECK-LABEL: kgen.generator @impl1_thunk<size, ty: dtype -> xyz>(%arg0: !meta.buffer<size, ty>) -> index
// CHECK-NEXT:  constraints <eq(:dtype ty, f32), "argument #0 specifies 'ty' = f32">
// CHECK-NEXT:  implements @bufitf {
// CHECK-NEXT:    %0 = meta.buffer.rebind %arg0 : !meta.buffer<size, ty> to !meta.buffer<size, f32>
// CHECK-NEXT:    %1 = kgen.call @impl1<size = size, ty: dtype = ty -> xyz>(%0) : (!meta.buffer<size, f32>) -> index
// CHECK-NEXT:    kgen.return <xyz = xyz> %1 : index
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.generator @impl1<size, ty: dtype -> xyz>(%arg0: !meta.buffer<size, f32>) -> index
// CHECK-NEXT: constraints <eq(:dtype ty, f32), "argument #0 specifies 'ty' = f32">
// CHECK: %0 = meta.buffer.size %arg0 : !meta.buffer<size, f32>

//===----------------------------------------------------------------------===//
// Simplify constraints.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @SetIntersect<
hlkgen.generator @SetIntersect<a, b>()
// CHECK-NEXT: constraints <
// CHECK-NEXT: in(a, [8, 57]), "thing 1, and thing two",
// CHECK-NEXT: in(b, [7, 8]), "thing Y"
  constraints <
    in(a, [7, 8, 57]), "thing 1",
    in(a, [57, 8, 2]), "thing two",

    in(b, [7, 8, 57]), "thing X",  // superset of B.
    in(b, [7, 8]), "thing Y"
   > {
  kgen.return
}
