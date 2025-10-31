// RUN: not kgen-opt -outline-closures-new=debug-build=true -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: value is a capture of closure fn but is not in the capture list

// Provide a struct generator entry for the closure under test
kgen.struct.generator @"closure_types::fn"<CAPTURES: !kgen.param_closure<@"closure_types" "fn">> = !kgen.closure<@"closure_types", "fn" nonescaping> {
}

kgen.generator @closure_types(%arg0 : index, %arg1 : index) {
  // CHECK: error: value is a capture of closure fn but is not in the capture list
  %0 = index.add %arg0, %arg0
  %3 = kgen.closure.init(%arg1)(%arg2: index) -> index {
    %1 = index.add %0, %arg0
    kgen.return %1 : index
  } : (index), !kgen.pointer<!kgen.closure<@closure_types, "fn" nonescaping>>
  kgen.return
}

// -----

// COM: Ensure that unhandled symbols and types fail the pass.

// CHECK: no type found for closure type '!kgen.closure<@invalid, "UNKNOWN" nonescaping>'

#type_value = #kgen.type<typevalue<#kgen.genref<@"invalid::fn">>, !kgen.closure<@"invalid", "UNKNOWN" nonescaping>> : !kgen.type

kgen.struct.generator @"invalid::fn"<CAPTURES: !kgen.param_closure<@"invalid" "fn">> = !kgen.closure<@"invalid", "fn" nonescaping> {
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"invalid", "UNKNOWN" nonescaping>> read_mem, index) -> index = #kgen.closure.symbol<@"invalid", "UNKNOWN", #kgen.closure_method<call>, <:!kgen.param_closure<@"invalid" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"invalid", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"invalid", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"invalid", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"invalid" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"invalid", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"invalid", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"invalid" "fn"> CAPTURES>>
  }
}

kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
  kgen.param.declare call: <index>(!kgen.pointer<x> read_mem) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: call](%arg0)
  kgen.return %0 : index
}


kgen.generator @invalid() {
  %3 = kgen.closure.init()(%arg2: index) -> index {
    kgen.return %arg2 : index
  } : (), !kgen.pointer<!kgen.closure<@invalid, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@invalid, "fn" nonescaping>> read_mem) -> index
  kgen.return
}
