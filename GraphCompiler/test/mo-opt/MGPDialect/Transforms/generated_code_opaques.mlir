// REQUIRES: disabled
// RUN: mo-opt --jit="dry-run=true dump-stub=true" %s -o %t 2>&1 | FileCheck %s


module {
  kgen.extern.generator  @"MOGGTests::MyCustomInt::__init__(MOGGTests::MyCustomInt=&,::Int)"(index, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none
  kgen.extern.generator  @"MOGGTests::MyCustomInt::__moveinit__(MOGGTests::MyCustomInt=&,MOGGTests::MyCustomInt)"(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none
  kgen.extern.generator  @"MOGGTests::MyCustomInt::__del__(MOGGTests::MyCustomInt)"(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none
  mgp.model @graph : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %3 = mgp.runtime.create %0, %1[1]
    mef.output %3, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.opaque<"">) {
    %0:2 = mogg.routine.async (%arg0) : (%arg2: !mef.chain) -> (!mef.chain, !mgp.opaque<"">){
      // %1 = mogg.mgp_to_async_ref(%arg2) : (!mef.chain) -> !kgen.pointer<none>
      %3 = mogg.stack_allocate_opaques : !mgp.opaque<"">
      %4 = "mogg.to_kgen"(%3) {typeMethodsDict = {__del__ = #kgen.symbol.constant<@"MOGGTests::MyCustomInt::__del__(MOGGTests::MyCustomInt)"> : !kgen.generator<(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>, __moveinit__ = #kgen.symbol.constant<@"MOGGTests::MyCustomInt::__moveinit__(MOGGTests::MyCustomInt=&,MOGGTests::MyCustomInt)"> : !kgen.generator<(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none>}} : (!mgp.opaque<"">) -> !kgen.pointer<struct<(index) memoryOnly>>
      %index42 = kgen.param.constant = <42>
      %5 = kgen.call @"MOGGTests::MyCustomInt::__init__(MOGGTests::MyCustomInt=&,::Int)"(%index42, %4) : (index, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none
      %6 = mogg.to_mgp_from_opaque(%4) : (!kgen.pointer<struct<(index) memoryOnly>>) -> !mgp.opaque<"">
      %7 = mogg.allocate_async_ref_for(%6) : (!mgp.opaque<"">) -> !kgen.pointer<none>
      mogg.create_async_mojo_value(%6, %7) : (!mgp.opaque<"">, !kgen.pointer<none>) -> ()
      %8 = mogg.async_ref_to_mgp(%7) : (!kgen.pointer<none>) -> !mgp.opaque<"">
      mogg.output %8 : !mgp.opaque<"">
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"custom__test_make_custom_int">> canFuseIntoOther = false inOutSignature = <"i"> {device = #M.device_ref<"cpu", 0>}
    mef.output %arg0, %0#1 : !mef.chain, !mgp.opaque<"">
  }
}


// CHECK-LABEL: kgen.generator export @__stub_
//  CHECK-SAME: ([[ARG0:%.*]]: !kgen.pointer<pointer<none>>, [[ARG1:%.*]]: !kgen.pointer<none>) {

// COM: Load the async value pointer.
// CHECK-NEXT: [[INDEX1:%.*]] = kgen.param.constant = <1>
// CHECK-NEXT: [[V0:%.*]] = pop.offset [[ARG0]][[[INDEX1]]] : !kgen.pointer<pointer<none>>

// COM: Load the async value pointer.
// CHECK-NEXT: [[INDEX0:%.*]] = kgen.param.constant = <0>
// CHECK-NEXT: [[TMP:%.*]] = pop.offset [[V0]][[[INDEX0]]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT: [[V1:%.*]] = pop.load [[TMP]] align<8> : !kgen.pointer<pointer<none>>

// COM: Do the stack allocation for the custom kernel output.
// CHECK: [[V2:%.*]] = pop.stack_allocation 1 x struct<(index) memoryOnly>

// COM: Call the custom kernel that creates the custom mojo value.
// CHECK-NEXT: [[INDEX42:%.*]] = kgen.param.constant = <42>
// CHECK: [[V4:%.*]] = kgen.call @"MOGGTests::MyCustomInt::__init__(MOGGTests::MyCustomInt=&,::[[INT:[A-Za-z0-9_]*]])"([[INDEX42]], {{.*}})

// COM: Get the move / del function pointers
// CHECK-NEXT: [[V5:%.*]] = kgen.create_reg_stub [(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none: @"MOGGTests::MyCustomInt::__moveinit__(MOGGTests::MyCustomInt=&,MOGGTests::MyCustomInt)"] : <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none>
// CHECK-NEXT: [[V6:%.*]] = kgen.create_reg_stub [(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none: @"MOGGTests::MyCustomInt::__del__(MOGGTests::MyCustomInt)"] : <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>

// COM: Get the sizeof and alignof properties needed to allocate heap memory for the mojo value.
// CHECK-NEXT: [[INDEX:%.*]] = kgen.param.constant = <get_sizeof(struct<(index) memoryOnly>, current_target())>
// CHECK-NEXT: [[INDEX_0:%.*]] = kgen.param.constant = <get_alignof(struct<(index) memoryOnly>, current_target())>

// COM: bitcast mov / del the expected type for create_mojo_value_async.
// CHECK: [[V7:%.*]] = pop.pointer.bitcast [[V5]] : !kgen.generator<(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none> to !kgen.generator<(!kgen.pointer<none>, !kgen.pointer<none>) -> !kgen.none>
// CHECK: [[V8:%.*]] = pop.pointer.bitcast [[V6]] : !kgen.generator<(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> to !kgen.generator<(!kgen.pointer<none>) -> !kgen.none>

// COM: call create_mojo_value_async
// CHECK: [[V12:%.*]] = kgen.call @"{{.*}}MOGGPrimitives::create_mojo_value_async({{.*}}, {{.*}}, [[INDEX]], [[INDEX_0]], [[V8]], [[V7]]) : (!kgen.pointer<none>, !kgen.pointer<none>, index, index, !kgen.generator<(!kgen.pointer<none>) -> !kgen.none>, !kgen.generator<(!kgen.pointer<none>, !kgen.pointer<none>) -> !kgen.none>) -> !kgen.none
// CHECK: hlcf.return
