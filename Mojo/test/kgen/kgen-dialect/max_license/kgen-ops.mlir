// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters -kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters -kgen-print-inline-type-values | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
kgen.generator @trivial_generator(%arg0: si32) -> si32 {
  // CHECK-NEXT: hlcf.return %[[ARG0]] : si32
  hlcf.return %arg0 : si32
}

// CHECK-LABEL: kgen.func @trivial
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
kgen.func @trivial(%arg0: si32) -> si32 {
  // CHECK-NEXT: hlcf.return %[[ARG0]] : si32
  hlcf.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @generator_with_source_name
// CHECk-SAME: attributes {sourceName = "some_func"}
kgen.generator @generator_with_source_name(%arg0: si32) -> si32 attributes {sourceName = "some_func"} {
  hlcf.return %arg0 : si32
}

// This checks the generic (also ugly) parameter syntax.
// CHECK-LABEL: kgen.generator @generic_parameter_syntax<p1, p2, ty: dtype>() {
kgen.generator @generic_parameter_syntax<p1, p2: index, ty: dtype>() {
  // CHECK: kgen.param.declare somei8: i8 = <5>
  kgen.param.declare somei8: i8 = <5>

  // Binding with pretty type 'dtype'
  // CHECK: kgen.param.declare someTy: dtype = <ty>
  kgen.param.declare someTy: dtype = <ty>

  // CHECK: kgen.param.declare p2a = <to_builtin(:scalar<index> add(from_builtin(p2), 1))>
  kgen.param.declare p2a = <add(p2, 1)>

  // CHECK: kgen.param.declare index: type = <index>
  kgen.param.declare index: type = <index>
  // CHECK: kgen.param.declare index_ref: type = <*"index">
  kgen.param.declare index_ref: type = <*"index">

  hlcf.return   // CHECK: hlcf.return
}

// CHECK-LABEL: kgen.generator @generate(
// CHECK-SAME: %[[ARG0:.*]]: si32
kgen.generator @generate(%arg0: si32) -> si32{
  // CHECK: %[[V0:.*]] = kgen.call @trivial_generator(%[[ARG0]]) : (si32) -> si32
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32

  // CHECK: %[[V1:.*]] = kgen.call @trivial(%[[V0]]) : (si32) -> si32
  %1 = kgen.call @trivial(%0) : (si32) -> si32

  hlcf.return %1 : si32
}

// CHECK-LABEL: kgen.generator @param_value<p1, p2: scalar<bool>>() -> (index, index)
kgen.generator @param_value<p1, p2: !kgen.scalar<bool>>() -> (index, index) {
  // CHECK: %[[V0:.*]] = kgen.param.constant = <p1>
  %0 = kgen.param.constant = <p1>

  // CHECK: %[[V1:.*]] = kgen.param.constant = <p1>
  %1 = kgen.param.constant: index = <p1>

  // CHECK: = kgen.param.constant: scalar<bool> = <p2>
  %2 = kgen.param.constant: scalar<bool> = <p2>

  // Param assert
  // CHECK: kgen.param.assert <p2>, "p2 is really important to be true"
  kgen.param.assert <p2>, "p2 is really important to be true"

  // CHECK: kgen.param.assert <identical(:scalar<index> from_builtin(p1), 42)>, "p1 seems like 42 is best"
  kgen.param.assert <eq(p1, 42)>, "p1 seems like 42 is best"

  // CHECK: hlcf.return %[[V0]], %[[V1]] : index, index
  hlcf.return %0, %1 : index, index
}

// CHECK-LABEL: kgen.generator @str_param_value<p, msg: string>() -> index
kgen.generator @str_param_value<p, msg: string>() -> index {
  // CHECK: %[[V:.*]] = kgen.param.constant = <p>
  %0 = kgen.param.constant = <p>

  // CHECK: kgen.param.assert <identical(:scalar<index> from_builtin(p), 42)>, msg
  kgen.param.assert <eq(p, 42)>, msg

  // CHECK: hlcf.return %[[V]] : index
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.generator @scalar_type<dt: dtype>()
kgen.generator @scalar_type<dt: !kgen.dtype>() {
  // CHECK: kgen.param.constant: dtype = <f32>
  kgen.param.constant: !kgen.dtype = <#kgen.dtype.constant<f32>>

  // CHECK: "test.someop"() : () -> !kgen.scalar<f32>
  %x = "test.someop"() {} : () -> !kgen.scalar<f32>
  // CHECK: "test.someop"() : () -> !kgen.scalar<f32>
  %y = "test.someop"() {} : () -> !kgen.scalar<#kgen.dtype.constant<f32> : !kgen.dtype>

  hlcf.return
}

// CHECK-LABEL: @scalar_params_verbose<T: dtype>
// CHECK-SAME: (%[[ARG0:.*]]:
kgen.generator @scalar_params_verbose<T: dtype>(%arg0:
           // CHECK-SAME: !kgen.scalar<T>) ->
           !kgen.scalar<#kgen.param.decl.ref<"T"> : !kgen.dtype>)
            // CHECK-SAME: !kgen.scalar<T> {
            -> !kgen.scalar<T> {
  // CHECK: hlcf.return %[[ARG0]] : !kgen.scalar<T>
  hlcf.return %arg0 : !kgen.scalar<#kgen.param.decl.ref<"T"> : !kgen.dtype>
}

// CHECK-LABEL: kgen.func @simple
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
kgen.func @simple(%arg0 : si32) -> si32 {

  // CHECK: %[[V0:.*]] = kgen.call @trivial_generator(%[[ARG0]]) : (si32) -> si32
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32

  // CHECK: hlcf.return %[[V0]] : si32
  hlcf.return %0 : si32
}

// Tests that the parameter verifier only runs KGENDialect
kgen.generator @test_getelementptr(%value: f32, %ptr: !llvm.ptr, %idx: i64) -> () {
  %element_ptr = llvm.getelementptr %ptr[%idx]: (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %value, %element_ptr : f32, !llvm.ptr
  hlcf.return
}


// CHECK-LABEL: kgen.generator @calleeRemap
// Caller/callee mapping.
kgen.generator @calleeRemap<type: dtype, size>(%x: !kgen.simd<size, type>) {
  hlcf.return
}

// CHECK-LABEL: kgen.generator @callerRemap
kgen.generator @callerRemap<ty : dtype, sz>(
    %arg0: !kgen.simd<4, ty>, %arg1: !kgen.simd<8, f64>, %arg2: !kgen.simd<sz, f32>) {
  kgen.call @calleeRemap<:dtype ty, 4>(%arg0) : (!kgen.simd<4, ty>) -> ()
  kgen.call @calleeRemap<:dtype f64, 8>(%arg1) : (!kgen.simd<8, f64>) -> ()
  kgen.call @calleeRemap<:dtype f32, sz>(%arg2) : (!kgen.simd<sz, f32>) -> ()
  hlcf.return
}

// CHECK-LABEL: kgen.generator @call_param
kgen.generator @call_param
  <dt: dtype,
   fn1: <dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)>,
   fn2: (si32) -> si32,
   fn3: <dtype>()->(),
   fn4: <type>(!kgen.param<*(0,0)>)->()
  >(%arg0: si32, %arg1: !kgen.simd<42, dt>) {

  %one = kgen.param.constant: scalar<si64> = <<1>>
  // CHECK: %[[ONE:.*]] = pop.cast
  %cst = pop.cast %one : !kgen.scalar<si64> to !kgen.scalar<dt>

  // CHECK: kgen.call_param[(!kgen.scalar<dt>) -> !kgen.scalar<dt>:
  // CHECK-SAME: bind_params(:<dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)> fn1, :dtype dt)](%[[ONE]])
  %0 = kgen.call_param[(!kgen.scalar<dt>) -> !kgen.scalar<dt>:
    bind_params(:<dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)> fn1, :dtype dt)](%cst)

  // Can pass an arbitrary expression, even a symbol reference (but this is
  // silly).
  // CHECK: kgen.call_param[(si32) -> si32: @trivial](%arg0)
  %1 = kgen.call_param[(si32) -> si32: @trivial](%arg0)

  // Can pass and receive parameters.
  // CHECK: kgen.call_param[() -> (): bind_params(:<dtype>() -> () fn3, :dtype f32)]()
  kgen.call_param[()->(): bind_params(:<dtype>()->() fn3, :dtype f32)]()
  kgen.param.declare resultIndex = <1>
  // CHECK: kgen.param.declare idx2 = <to_builtin(:scalar<index> mul(from_builtin(resultIndex), 3))>
  kgen.param.declare idx2 = <add(resultIndex, resultIndex, resultIndex)>

  // Call a kgen.symbol.constant with unbound parameters.
  // CHECK: kgen.call_param[(!kgen.simd<42, dt>) -> (): @calleeRemap<:dtype dt, 42>](%arg1)
  kgen.call_param[(!kgen.simd<42, dt>) -> (): bind_params(:<dtype, index>(!kgen.simd<*(0,1), *(0,0)>)->() @calleeRemap, :dtype dt, :index 42)](%arg1)

  // Call a kgen.symbol.constant with parameters bound in symbol.constant.
  // CHECK: kgen.call_param[(!kgen.simd<42, dt>) -> ():
  // CHECK-SAME: @calleeRemap<:dtype dt, 42>](%arg1)
  kgen.call_param[(!kgen.simd<42, dt>)->():
                      #kgen.symbol.constant<@calleeRemap<:dtype dt, 42>>]
     (%arg1)

  hlcf.return
}

// CHECK-LABEL: @rebind
kgen.generator @rebind<type1: type, type2: type, dtype1: dtype, dtype2: dtype>(%a: !kgen.param<type1>) {
  // CHECK: kgen.rebind %{{.*}} : !kgen.param<type1> to !kgen.param<type2>
  %0 = kgen.rebind %a : !kgen.param<type1> to !kgen.param<type2>
  // CHECK: kgen.rebind %{{.*}} : !kgen.param<type2> to !kgen.scalar<dtype1>
  %1 = kgen.rebind %0 : !kgen.param<type2> to !kgen.scalar<dtype1>
  // CHECK: kgen.rebind %{{.*}} : !kgen.scalar<dtype1> to !kgen.scalar<dtype2>
  %2 = kgen.rebind %1 : !kgen.scalar<dtype1> to !kgen.scalar<dtype2>
  // CHECK: kgen.rebind %{{.*}} : !kgen.scalar<dtype2> to !kgen.scalar<f32>
  %3 = kgen.rebind %2 : !kgen.scalar<dtype2> to !kgen.scalar<f32>
  // CHECK: kgen.rebind %{{.*}} : !kgen.scalar<f32> to !kgen.param<type1>
  %4 = kgen.rebind %3 : !kgen.scalar<f32> to !kgen.param<type1>
  // CHECK: kgen.rebind %{{.*}} : !kgen.param<type1> to i32
  %5 = kgen.rebind %a : !kgen.param<type1> to i32
  hlcf.return
}

// CHECK-LABEL: @param_func_0
kgen.generator @param_func_0() -> index {
  %0 = index.constant 0
  hlcf.return %0 : index
}

// CHECK-LABEL: @param_func_1
kgen.generator @param_func_1() -> index {
  %0 = index.constant 0
  hlcf.return %0 : index
}

// CHECK-LABEL: @constexpr_rebind
kgen.generator @constexpr_rebind(%arg0: !pop.array<apply(:() -> index @param_func_0), index>) {
  // CHECK-LABEL: kgen.rebind %arg0
  %0 = kgen.rebind %arg0 : !pop.array<apply(:() -> index @param_func_0), index> to
                           !pop.array<apply(:() -> index @param_func_1), index>
  hlcf.return
}

lit.struct.decl @Foo<N> {}

// CHECK-LABEL: @rebind_ref
kgen.generator @rebind_ref<N>(%arg0: !lit.struct<@Foo<N>>) {
  // CHECK: kgen.rebind %arg0 : !lit.struct<@Foo<N>> to !lit.struct<@Foo<1>>
  %0 = kgen.rebind %arg0 : !lit.struct<@Foo<N>> to !lit.struct<@Foo<1>>
  hlcf.return
}

// CHECK-LABEL: @variant_type
kgen.generator @variant_type<N, T: type>(%a: !kgen.simd<N, f32>) -> !kgen.param<T> {
  // CHECK: kgen.variant.create %arg0, 1 : <T, simd<N, f32>>
  %0 = kgen.variant.create %a, 1 : <T, simd<N, f32>>
  // CHECK: kgen.variant.is %0, 0 : <T, simd<N, f32>>
  %1 = kgen.variant.is %0, 0 : <T, simd<N, f32>>
  // CHECK: kgen.variant.get %0, 0 : <T, simd<N, f32>>
  %2 = kgen.variant.get %0, 0 : <T, simd<N, f32>>
  hlcf.return %2 : !kgen.param<T>
}

// CHECK-LABEL: kgen.generator @nullary_generator
kgen.generator @nullary_generator() {
  hlcf.return
}

// CHECK-LABEL: kgen.generator @one_result_parametric_generator
kgen.generator @one_result_parametric_generator<size>() -> index {
  %0 = kgen.param.constant = <size>
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.generator @signature_parametric_generator
kgen.generator @signature_parametric_generator<fn: () -> ()>() {
  hlcf.return
}

// CHECK-LABEL: @DependentParam
// CHECK-SAME: <DT: dtype, V: scalar<DT>>
lit.struct.decl @DependentParam<DT: dtype, V: scalar<DT>> {}

// CHECK-LABEL: @specialize_struct_decl
// CHECK-SAME: @DependentParam<:dtype f32, :scalar<f32> *?>>
kgen.generator @specialize_struct_decl(%p: !lit.struct<@DependentParam<:dtype f32, :scalar<f32> *?>>) {
  hlcf.return
}

// CHECK-LABEL: @genWithConventions0({{.*}}!kgen.pointer<i32> owned) {
kgen.generator @genWithConventions0(%arg0: !kgen.pointer<i32> owned, %arg1: !kgen.pointer<i32> owned) {
  hlcf.return
}

// CHECK-LABEL: @genWithConventions1({{.*}}!kgen.pointer<i32> mut) {
kgen.generator @genWithConventions1(%a: !kgen.pointer<i32>owned , %b: !kgen.pointer<i32> mut) {
  hlcf.return
}

// CHECK-LABEL: @genWithConventions2(%arg0: !kgen.pointer<i32> mut, %arg1: !kgen.pointer<i32> owned)
kgen.generator @genWithConventions2(%a: !kgen.pointer<i32> mut, %b: !kgen.pointer<i32> owned) {
  hlcf.return
}

// CHECK-LABEL: @genWithConventions3
// CHECK-SAME: pointer<i32> owned_in_mem,
// CHECK-SAME: pointer<i32> byref_result) -> !kgen.none
kgen.generator @genWithConventions3(
    %a: !kgen.pointer<i32> owned_in_mem,
    %r: !kgen.pointer<i32> byref_result) -> !kgen.none {
  // CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
  %0 = kgen.param.constant: none = <#kgen.none>
  hlcf.return %0 : !kgen.none
}

// CHECK-LABEL: lit.struct.decl @Error
lit.struct.decl @Error {}

// CHECK-LABEL: kgen.generator @signature_types_with_convention
kgen.generator @signature_types_with_convention<
    // CHECK-SAME: fn1: <index>(!kgen.pointer<i32>) -> (),
    fn1: <index>(!kgen.pointer<i32>) -> (),
    // CHECK-SAME: fn2: (!kgen.pointer<i32>) throws -> !kgen.variant<!lit.struct<@Error>, i32>,
    fn2: (!kgen.pointer<i32>) throws -> !kgen.variant<!lit.struct<@Error>, i32>,
    // CHECK-SAME: fn3: (!kgen.pointer<i32> mut) -> ()
    fn3: (!kgen.pointer<i32> mut) -> ()
>(%arg0: !kgen.pointer<i32>) {
  // CHECK: kgen.call @genWithConventions2(%arg0, %arg0) : (!kgen.pointer<i32> mut, !kgen.pointer<i32> owned) -> ()
  kgen.call @genWithConventions2(%arg0, %arg0) : (!kgen.pointer<i32> mut, !kgen.pointer<i32> owned) -> ()
  hlcf.return
}

// CHECK-LABEL: kgen.generator @forceInline() always_inline
kgen.generator @forceInline() always_inline {
  hlcf.return
}

// CHECK-LABEL: kgen.generator @forceInlineAndThrows() throws -> !kgen.variant<!lit.struct<@Error>, index> always_inline
kgen.generator @forceInlineAndThrows() throws -> !kgen.variant<!lit.struct<@Error>, index> always_inline {
  %0 = index.constant 0
  %1 = kgen.variant.create %0, 1 : <!lit.struct<@Error>, index>
  hlcf.return %1 : !kgen.variant<!lit.struct<@Error>, index>
}

// CHECK-LABEL: @async_generator
// CHECK-SAME <A>() async -> index
kgen.generator @async_generator<A>() async -> index {
  %0 = kgen.param.constant = <A>
  hlcf.return %0 : index
}

// CHECK-LABEL: @call_async_generator
kgen.generator @call_async_generator() {
  // CHECK-NEXT: call @async_generator<1>() : () async -> index
  %0 = kgen.call @async_generator<1>() : () async -> index
  hlcf.return
}

// CHECK-LABEL: @constexprIf()
kgen.generator @constexprIf() {
  // CHECK-NEXT: param.declare cond: scalar<bool> = <false>
  kgen.param.declare cond: scalar<bool> = <false>
  // CHECK-NEXT: comptime.if cond {
  hlcf.comptime.if cond {
    // CHECK-NEXT: param.declare true = <32>
    kgen.param.declare true = <32>
    // CHECK-NEXT: return
    hlcf.return
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: param.declare false = <35>
    kgen.param.declare false = <35>
    // CHECK-NEXT: yield
    hlcf.comptime.yield
  }
  hlcf.return
}

// CHECK-LABEL: @comptimeIfElif()
kgen.generator @comptimeIfElif() -> index {
  // CHECK-NEXT: %[[R:.*]] = hlcf.comptime.if false -> index {
  // CHECK-NEXT:   %[[A:.*]] = kgen.param.constant = <1>
  // CHECK-NEXT:   hlcf.comptime.yield %[[A]] : index
  // CHECK-NEXT: } elif true {
  // CHECK-NEXT:   %[[B:.*]] = kgen.param.constant = <2>
  // CHECK-NEXT:   hlcf.comptime.yield %[[B]] : index
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[C:.*]] = kgen.param.constant = <3>
  // CHECK-NEXT:   hlcf.comptime.yield %[[C]] : index
  // CHECK-NEXT: }
  %0 = hlcf.comptime.if false -> index {
    %a = kgen.param.constant = <1>
    hlcf.comptime.yield %a : index
  } elif true {
    %b = kgen.param.constant = <2>
    hlcf.comptime.yield %b : index
  } else {
    %c = kgen.param.constant = <3>
    hlcf.comptime.yield %c : index
  }
  // CHECK-NEXT: hlcf.return %[[R]] : index
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.generator @always_dead_function
kgen.generator @always_dead_function() -> index {
  // CHECK-NEXT: %index = kgen.param.constant = <*?>
  %0 = kgen.param.constant = <*?>
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.generator @param_declare_dependent
kgen.generator @param_declare_dependent<fn: <index>(!pop.array<*(0,0), i1>) -> ()>() {
  // CHECK: ref: <index>(!pop.array<*(0,0), i1>) -> () = <fn>
  kgen.param.declare ref: <index>(!pop.array<*(0,0), i1>) -> () = <fn>
  // CHECK: bound: (!pop.array<2, i1>) -> () =
  kgen.param.declare bound: (!pop.array<2, i1>) -> () =
    <bind_params(:<index>(!pop.array<*(0,0), i1>) -> () ref, 2)>
  hlcf.return
}

kgen.func @closure1(%arg0: index) -> index {
  %idx4 = index.constant 4
  hlcf.return %idx4 : index
}

kgen.func @closure2(%arg0: index, %arg1: index) -> index {
  %idx4 = index.constant 4
  hlcf.return %idx4 : index
}

// CHECK-LABEL: kgen.generator @create_create_func_ptr
kgen.generator @create_create_func_ptr<fn: (index) -> index>(%arg0: index) {
  // CHECK: kgen.create_func_ptr[(index) -> index: @closure1]()
  %0 = kgen.create_func_ptr[(index) -> index: @closure1]()
  // CHECK: kgen.create_func_ptr[(index, index) -> index: @closure2]()
  %1 = kgen.create_func_ptr[(index, index) -> index: @closure2]()

  // CHECK: kgen.create_func_ptr[(index) -> index: fn]()
  %2 = kgen.create_func_ptr[(index) -> index: fn]()
  // CHECK: call_indirect %2(%arg0) : (index) -> index
  kgen.call_indirect %2(%arg0) : (index) -> index

  hlcf.return
}

kgen.generator @evaluator(%fns: !kgen.pointer<(index) -> i1>, %sz: index) -> index {
  hlcf.unreachable
}

kgen.generator @fooImpl(%arg0: index) -> i1 {
  hlcf.unreachable
}

kgen.generator @barImpl(%arg0: index) -> i1 {
  hlcf.unreachable
}

// CHECK-LABEL: @cost_of_op
kgen.generator @cost_of_op() ->
  (index, index, index, index, index, index, index, index) {
  // CHECK-NEXT: kgen.cost_of[(index) -> i1: @fooImpl]
  %loads, %stores, %adds, %cmps, %divs, %muls, %mul_adds, %other =
    kgen.cost_of[(index) -> i1: @fooImpl]
  hlcf.return %loads, %stores, %adds, %cmps, %divs, %muls, %mul_adds, %other  :
    index, index, index, index, index, index, index, index
}

// CHECK: kgen.extern.generator @external_generator<width>(index) -> !kgen.simd<width, f32>
kgen.extern.generator @external_generator<width>(index) -> !kgen.simd<width, f32>

// CHECK-LABEL: kgen.generator @call_extern_funcs
kgen.generator @call_extern_funcs<width>(%0: index) -> !kgen.simd<width, f32> {
  // CHECK-NEXT: call @external_generator<width>(%{{.*}}) : (index) -> !kgen.simd<width, f32>
  %1 = kgen.call @external_generator<width>(%0) : (index) -> !kgen.simd<width, f32>
  hlcf.return %1 : !kgen.simd<width, f32>
}

#file = #debuginfo.file<"foo.mlir" in "/">
#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<i32>
#lexical_block = #debuginfo.lexical_block<scope = #subprogram, file = #file, line = 104, column = 17>
#lexical_block1 = #debuginfo.lexical_block<scope = #subprogram, file = #file, line = 120, column = 22>

#loc = loc("foo.mlir":7:8)
#loc1 = loc("bar.mlir":5:6)
#funcLoc = loc(fused<#subprogram>[#loc])
#callsite = loc(callsite(#loc1 at #funcLoc))

// CHECK-LABEL: @debuginfo_locs
kgen.func @debuginfo_locs() {
  %index0 = kgen.param.constant = <0> loc(callsite(#loc1 at #funcLoc))
  %index1 = kgen.param.constant = <1> loc(callsite(#loc1 at #callsite))
  %index2 = kgen.param.constant = <2> loc(fused<#lexical_block1>[#loc])
  hlcf.return loc(#funcLoc)
} loc(#funcLoc)

// COM: Check that named locations don't trip up verification.
// CHECK-LABEL: @wrappedLoc
kgen.generator @wrappedLoc() {
  hlcf.return
} loc("stuff"(callsite("foo.mlir":10:12 at "bar.mlir":17:18)))


// CHECK-LABEL: kgen.func export @exported_func
kgen.func export @exported_func() {
  hlcf.return
}

// CHECK-LABEL: kgen.func @source_loc
kgen.func @source_loc() {
  // CHECK: kgen.source_loc[0]
  %line, %col, %fileName = kgen.source_loc[0]
  // CHECK: kgen.source_loc[5]
  %line1, %col1, %fileName1 = kgen.source_loc[5]
  hlcf.return
}

// CHECK-LABEL: kgen.func @tailcall
kgen.func @tailcall(%arg0 : si32, %arg1: !kgen.generator<(si32) -> si32>) -> si32 {
  // CHECK: [[V0:%.*]] = kgen.call_indirect musttail %arg1(%arg0) : (si32) -> si32
  %0 = kgen.call_indirect musttail %arg1(%arg0) : (si32) -> si32
  // CHECK: [[V1:%.*]] = kgen.call_indirect %arg1(%arg0) : (si32) -> si32
  %1 = kgen.call_indirect %arg1(%arg0) : (si32) -> si32
  // CHECK: kgen.call tail @source_loc() : () -> ()
  kgen.call tail @source_loc() : () -> ()
  hlcf.return %0 : si32
}

// CHECK-LABEL: kgen.struct.generator @LinkedList<T: type> = struct_inst<"LinkedList"[T]<:type T>
// CHECK-SAME: (data: T, next: [typevalue<#kgen.genref<@LinkedList<:type T>>>, pointer<none>])>
kgen.struct.generator @LinkedList<T: type> =
  struct_inst<"LinkedList"[T]<:type T>(
    data: [typevalue<T>, !kgen.param<T>],
    next: [typevalue<#kgen.genref<@LinkedList<:type T>>>, pointer<none>]
  )>

// CHECK-LABEL: kgen.struct.generator @LinkedListVerbose<T: type> = struct_inst<"LinkedListVerbose"[T]<:type T>
// CHECK-SAME: (data: T, next: [typevalue<#kgen.genref<@LinkedListVerbose<:type T>>>, pointer<none>])>
kgen.struct.generator @LinkedListVerbose<T: type> : type =
  struct_inst<"LinkedListVerbose"[T]<:type T>(
    data: [typevalue<T>, !kgen.param<T>],
    next: [typevalue<#kgen.genref<@LinkedListVerbose<:type T>>>, pointer<none>]
  )> {}

kgen.func @"NonParametric::__bool__(::NonParametric)"(%arg0: !kgen.struct<()>) -> i1 {
  %index1 = kgen.param.constant : i1 = <1>
  hlcf.return %index1 : i1
}

// CHECK-LABEL: kgen.struct.generator @NonParametric = struct_inst<"NonParametric"(data: index)>
kgen.struct.generator @NonParametric = struct_inst<"NonParametric"(data: index)> {
  // CHECK: kgen.conformance @Boolable {
  kgen.conformance @Boolable {
    // CHECK: kgen.witness "__bool__" : (!kgen.struct<()>) -> i1 = @"NonParametric::__bool__(::NonParametric)"
    kgen.witness "__bool__" : (!kgen.struct<()>) -> i1 = @"NonParametric::__bool__(::NonParametric)"
  }
}

kgen.generator @"ParamStruct::__bool__(::ParamStruct)"<T: type>(%arg0: !kgen.struct<(T)>) -> i1 {
  %index1 = kgen.param.constant : i1 = <1>
  hlcf.return %index1 : i1
}

// CHECK-LABEL: kgen.struct.generator @ParamStruct<T: type> = struct_inst<"ParamStruct"[T]<:type T>(data: T)>
kgen.struct.generator @ParamStruct<T: type> = struct_inst<"ParamStruct"[T]<:type T>(data: T)> {
  // CHECK: kgen.conformance @Boolable {
  kgen.conformance @Boolable {
    // CHECK: kgen.witness "__bool__" : (!kgen.struct<(T)>) -> i1 = @"ParamStruct::__bool__(::ParamStruct)"<:type T>
    kgen.witness "__bool__" : (!kgen.struct<(T)>) -> i1 = @"ParamStruct::__bool__(::ParamStruct)"<:type T>
  }
}

// Test: ConformanceOp with constraint (conditional conformance)
// This tests the round-trip of the `where` syntax for conditional conformances.
kgen.generator @"ConditionalStruct::__copy__(::ConditionalStruct)"<T: type>(%arg0: !kgen.struct<(T)>) -> !kgen.struct<(T)> {
  hlcf.unreachable
}

#loc_conditional = loc("conditional.mojo":10:5)

// CHECK-LABEL: kgen.struct.generator @ConditionalStruct<T: type>
kgen.struct.generator @ConditionalStruct<T: type> = struct_inst<"ConditionalStruct"[T]<:type T>(data: [typevalue<T>, !kgen.param<T>])> {
  // CHECK: kgen.conformance @Copyable {
  kgen.conformance @Copyable {
    kgen.witness "__copy__" : (!kgen.struct<(T)>) -> !kgen.struct<(T)> = @"ConditionalStruct::__copy__(::ConditionalStruct)"<:type T>
  } where #kgen.constraint<conforms_to(:type T, :type #kgen.type<typevalue<#kgen.trait_ref<[@Copyable]>>, type> : !kgen.type), #loc_conditional>

  // Unconditional conformance for comparison
  // CHECK: kgen.conformance @Boolable {
  // CHECK-NOT: where #kgen.constraint
  kgen.conformance @Boolable {
    kgen.witness "__bool__" : (!kgen.struct<(T)>) -> i1 = @"ParamStruct::__bool__(::ParamStruct)"<:type T>
  }
}

// Test: ConformanceOp with complex constraint expression
#loc_complex = loc("complex.mojo":20:10)

// CHECK-LABEL: kgen.struct.generator @ComplexConditional<T: type, N>
kgen.struct.generator @ComplexConditional<T: type, N> = struct_inst<"ComplexConditional"[T, N]<:type T, N>(data: [typevalue<T>, !kgen.param<T>])> {
  // CHECK: kgen.conformance @Iterable {
  kgen.conformance @Iterable {
  } where #kgen.constraint<#kgen.param.expr<and, #kgen.type_conforms_to_trait<#kgen.param.decl.ref<"T"> : !kgen.type, #kgen.type<typevalue<#kgen.trait_ref<[@Sized]>>, type> : !kgen.type>, #kgen.param.expr<xor, #kgen.param.expr<lt, #kgen.param.decl.ref<"N"> : index, 0 : index> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>> : !kgen.scalar<bool>, #loc_complex>
}

// CHECK-LABEL: kgen.generator @unnamed_argument(
// CHECK-SAME: %[[ARG0:.*]]: si32,
// CHECK-SAME: %[[ARG1:.*]]: si32
kgen.generator @unnamed_argument(%arg0: si32, si32) -> si32 {
  // CHECK-NEXT: hlcf.return %[[ARG0]]
  hlcf.return %arg0 : si32
}

kgen.generator @kernel() {
  hlcf.return
}

kgen.generator @compile_offload() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>>

  // CHECK: %[[V0:.*]] = kgen.compile_offload<nvptx, 0, "", "", :() -> () @kernel> : !kgen.none
  %0 = kgen.compile_offload<nvptx, 0, "", "", : ()->() @kernel> : !kgen.none

  hlcf.return
}

// CHECK-LABEL: kgen.generator @compile_offload_with_link_option
kgen.generator @compile_offload_with_link_option() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>>

  // CHECK: %[[V0:.*]] = kgen.compile_offload<nvptx, 0, "", "some-link-flag", :() -> () @kernel> : !kgen.none
  %0 = kgen.compile_offload<nvptx, 0, "", "some-link-flag", : ()->() @kernel> : !kgen.none

  hlcf.return
}

// CHECK-LABEL: kgen.generator @compile_offload_with_target_attrs
kgen.generator @compile_offload_with_target_attrs() {
  kgen.param.declare tpu: target = <#kgen.target<triple = "tpu-google-xla", arch = "tpu-v6e", data_layout = "e-i64:64", simd_bit_width = 1024>>

  // CHECK: %[[V0:.*]] = kgen.compile_offload<tpu, 0, "", "", :() -> () @kernel {target_attrs = {tpu.dimension_semantics = [0, 1], tpu.grid_dims = [2, 4]}}> : !kgen.none
  %0 = kgen.compile_offload<tpu, 0, "", "", : ()->() @kernel {target_attrs = {"tpu.grid_dims" = [2, 4], "tpu.dimension_semantics" = [0, 1]}}> : !kgen.none

  hlcf.return
}

// CHECK-LABEL: kgen.generator @is_run_in_comptime_interpreter
kgen.generator @is_run_in_comptime_interpreter() -> index {
  %index1 = kgen.param.constant = <1>
  %index2 = kgen.param.constant = <2>
  // CHECK: kgen.is_run_in_comptime_interpreter
  %0 = kgen.is_run_in_comptime_interpreter : !kgen.scalar<bool>
  %3 = pop.select %0, %index1, %index2 : index
  hlcf.return %3 : index

}

// CHECK-LABEL: kgen.generator @codegen_reachable
kgen.generator @codegen_reachable<p: i1>() {

  // CHECK: kgen.codegen.reachable
  kgen.codegen.reachable<p>, "codegen_reachable_test"
  hlcf.return
}

kgen.generator @deferred_op(%arg0: index, %arg1: index) {
  // One operand.
  // CHECK: %[[V0:.*]] = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} : i1
  %0 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} : i1
  // Multiple operands.
  // CHECK: %[[V1:.*]] = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} : i1
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} : i1
  // No results.
  // CHECK: kgen.deferred "test.foo"(%arg0 : index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred}
  kgen.deferred "test.foo"(%arg0 : index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred}
  // With `opProperties`.
  // CHECK: kgen.deferred "test.foo"(%arg0 : index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} properties {operandSegmentSizes = array<i32: 1, 0>}
  kgen.deferred "test.foo"(%arg0 : index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} properties {operandSegmentSizes = array<i32: 1, 0>}
  hlcf.return
}

// CHECK-LABEL: kgen.generator @kgen_bytecode_extra_types
kgen.generator @kgen_bytecode_extra_types() {
  // CHECK: kgen.param.declare never_t: type = <!kgen.never>
  kgen.param.declare never_t: type = <!kgen.never>
  // CHECK: kgen.param.declare def_t: type = <!kgen.deferred>
  kgen.param.declare def_t: type = <!kgen.deferred>
  hlcf.return
}
