// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" | FileCheck %s

// In each #interp.memref blob tuple, the final list gives symbol-mark byte
// offsets: [0] marks a symbol starting at byte 0; [] means no symbol marks.

!pair = !kgen.struct<((index) -> index, i32)>

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:32:32">} {
  kgen.generator @target(%arg0: index) -> index {
    hlcf.return %arg0 : index
  }

  kgen.generator @store_function_then_adjacent_integer(%symbol: !kgen.generator<(index) -> index>) -> !kgen.pointer<i8> {
    %align = kgen.param.constant: scalar<index> = <4>
    %size = kgen.param.constant: scalar<index> = <8>
    %mem = pop.aligned_alloc %align, %size : !kgen.pointer<i8>
    %pair = pop.pointer.bitcast %mem : !kgen.pointer<i8> to !kgen.pointer<!pair>
    %function = kgen.struct.gep %pair[0] : <!pair>
    pop.store %symbol, %function : !kgen.pointer<(index) -> index>
    %integer = kgen.struct.gep %pair[1] : <!pair>
    %value = kgen.param.constant: i32 = <42>
    pop.store %value, %integer : !kgen.pointer<i32>
    hlcf.return %mem : !kgen.pointer<i8>
  }

  // Store a four-byte function, then adjacent i32 42; symbol offset 0 must remain.
  // CHECK-LABEL: kgen.func export @preserve_adjacent_function32
  kgen.generator export @preserve_adjacent_function32() -> !kgen.pointer<i8> {
    kgen.param.apply memory = [(!kgen.generator<(index) -> index>) -> !kgen.pointer<i8>: @store_function_then_adjacent_integer](@target)
    // CHECK: %pointer = kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [], [0])], [#kgen.gen<#kgen.func.symbol<@target>> : !kgen.generator<(index) -> index>]}, 0, 0>>
    %pointer = kgen.param.constant: pointer<i8> = <memory>
    hlcf.return %pointer : !kgen.pointer<i8>
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64">} {
  kgen.generator @target(%arg0: index) -> index {
    hlcf.return %arg0 : index
  }

  kgen.generator @store_function_then_integer_at_offset(%symbol: !kgen.generator<(index) -> index>, %offset: index) -> !kgen.pointer<i8> {
    %align = kgen.param.constant: scalar<index> = <8>
    %size = kgen.param.constant: scalar<index> = <16>
    %mem = pop.aligned_alloc %align, %size : !kgen.pointer<i8>
    %function = pop.pointer.bitcast %mem : !kgen.pointer<i8> to !kgen.pointer<(index) -> index>
    pop.store %symbol, %function : !kgen.pointer<(index) -> index>
    %offidx = pop.cast_from_builtin %offset : index to !kgen.scalar<index>
    %tail = pop.offset %mem[%offidx] : !kgen.pointer<i8>
    %integer = pop.pointer.bitcast %tail : !kgen.pointer<i8> to !kgen.pointer<i32>
    %value = kgen.param.constant: i32 = <42>
    pop.store %value, %integer : !kgen.pointer<i32>
    hlcf.return %mem : !kgen.pointer<i8>
  }

  // CHECK-LABEL: kgen.func export @overwrite_function64
  kgen.generator export @overwrite_function64() -> (!kgen.pointer<i8>, !kgen.pointer<i8>, !kgen.pointer<i8>) {
    // Store i32 42 at byte 3 of an eight-byte function; no symbol mark may remain.
    kgen.param.apply unaligned = [(!kgen.generator<(index) -> index>, index) -> !kgen.pointer<i8>: @store_function_then_integer_at_offset](@target, 3)
    // CHECK: kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [], [])],
    %unaligned = kgen.param.constant: pointer<i8> = <unaligned>
    // Store i32 42 over the function's last byte and beyond; no mark may remain.
    kgen.param.apply tail = [(!kgen.generator<(index) -> index>, index) -> !kgen.pointer<i8>: @store_function_then_integer_at_offset](@target, 7)
    // CHECK-NEXT: kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [], [])],
    %tail = kgen.param.constant: pointer<i8> = <tail>
    // Store i32 42 just after the function's eight bytes; offset 0 must remain.
    kgen.param.apply adjacent = [(!kgen.generator<(index) -> index>, index) -> !kgen.pointer<i8>: @store_function_then_integer_at_offset](@target, 8)
    // CHECK-NEXT: kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [], [0])],
    %adjacent = kgen.param.constant: pointer<i8> = <adjacent>
    hlcf.return %unaligned, %tail, %adjacent : !kgen.pointer<i8>, !kgen.pointer<i8>, !kgen.pointer<i8>
  }

  kgen.generator @store_pointer_then_function(%symbol: !kgen.generator<(index) -> index>, %offset: index, %free_pointee: !kgen.scalar<bool>) -> !kgen.pointer<i8> {
    %align = kgen.param.constant: scalar<index> = <8>
    %size = kgen.param.constant: scalar<index> = <16>
    %mem = pop.aligned_alloc %align, %size : !kgen.pointer<i8>
    %inner = pop.aligned_alloc %align, %size : !kgen.pointer<i8>
    %slot = pop.pointer.bitcast %mem : !kgen.pointer<i8> to !kgen.pointer<!kgen.pointer<i8>>
    pop.store %inner, %slot : !kgen.pointer<!kgen.pointer<i8>>
    hlcf.if %free_pointee {
      pop.aligned_free %inner : !kgen.pointer<i8>
      hlcf.yield
    } else {
      hlcf.yield
    }
    %offidx = pop.cast_from_builtin %offset : index to !kgen.scalar<index>
    %at = pop.offset %mem[%offidx] : !kgen.pointer<i8>
    %function = pop.pointer.bitcast %at : !kgen.pointer<i8> to !kgen.pointer<(index) -> index>
    pop.store %symbol, %function : !kgen.pointer<(index) -> index>
    hlcf.return %mem : !kgen.pointer<i8>
  }

  kgen.generator @store_pointer_function_pointer(%symbol: !kgen.generator<(index) -> index>) -> !kgen.pointer<i8> {
    %zero = index.constant 0
    %free_pointee = kgen.param.constant: scalar<bool> = <false>
    %mem = kgen.call @store_pointer_then_function(%symbol, %zero, %free_pointee) : (!kgen.generator<(index) -> index>, index, !kgen.scalar<bool>) -> !kgen.pointer<i8>
    // A partial pointer store exposes the old bit that export would discard.
    %four = kgen.param.constant: scalar<index> = <4>
    %at = pop.offset %mem[%four] : !kgen.pointer<i8>
    %slot = pop.pointer.bitcast %at : !kgen.pointer<i8> to !kgen.pointer<!kgen.pointer<i8>>
    pop.store %mem, %slot : !kgen.pointer<!kgen.pointer<i8>>
    hlcf.return %mem : !kgen.pointer<i8>
  }

  // CHECK-LABEL: kgen.func export @overwrite_pointer_with_function64
  kgen.generator export @overwrite_pointer_with_function64() -> (!kgen.pointer<i8>, !kgen.pointer<i8>) {
    kgen.param.apply exact = [(!kgen.generator<(index) -> index>) -> !kgen.pointer<i8>: @store_pointer_function_pointer](@target)
    // CHECK: kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [(4, 0, 0)], [])
    %exact = kgen.param.constant: pointer<i8> = <exact>
    kgen.param.apply stale = [(!kgen.generator<(index) -> index>, index, !kgen.scalar<bool>) -> !kgen.pointer<i8>: @store_pointer_then_function](@target, 4, true)
    // CHECK-NEXT: kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [], [4])
    %stale = kgen.param.constant: pointer<i8> = <stale>
    hlcf.return %exact, %stale : !kgen.pointer<i8>, !kgen.pointer<i8>
  }

  kgen.generator @store_pointer_over_stale_pointer() -> !kgen.pointer<i8> {
    %align = kgen.param.constant: scalar<index> = <8>
    %size = kgen.param.constant: scalar<index> = <16>
    %mem = pop.aligned_alloc %align, %size : !kgen.pointer<i8>
    %inner = pop.aligned_alloc %align, %size : !kgen.pointer<i8>
    %slot = pop.pointer.bitcast %mem : !kgen.pointer<i8> to !kgen.pointer<!kgen.pointer<i8>>
    pop.store %inner, %slot : !kgen.pointer<!kgen.pointer<i8>>
    pop.aligned_free %inner : !kgen.pointer<i8>
    %four = kgen.param.constant: scalar<index> = <4>
    %at = pop.offset %mem[%four] : !kgen.pointer<i8>
    %slot4 = pop.pointer.bitcast %at : !kgen.pointer<i8> to !kgen.pointer<!kgen.pointer<i8>>
    pop.store %mem, %slot4 : !kgen.pointer<!kgen.pointer<i8>>
    hlcf.return %mem : !kgen.pointer<i8>
  }

  // CHECK-LABEL: kgen.func export @overwrite_stale_pointer_with_pointer64
  kgen.generator export @overwrite_stale_pointer_with_pointer64() -> !kgen.pointer<i8> {
    kgen.param.apply memory = [() -> !kgen.pointer<i8>: @store_pointer_over_stale_pointer]()
    // CHECK: kgen.param.constant: pointer<i8> = <#interp.memref<{[({{.*}}, heap, [(4, 0, 0)], [])
    %memory = kgen.param.constant: pointer<i8> = <memory>
    hlcf.return %memory : !kgen.pointer<i8>
  }
}
