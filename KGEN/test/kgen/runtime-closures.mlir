// RUN: kgen %s -execute -func="main_capturing_region:index()" -func="closure_arg:index()" | FileCheck %s

kgen.func @capturing_region(%arg0: index) -> index {
    %idx4 = index.constant 4
    %0 = kgen.stage_closure = (%arg1: index) capturing -> index {
        %5 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
        %6 = pop.cast_from_builtin %arg1 : index to !pop.scalar<index>
        %7 = pop.add %5, %6 : !pop.scalar<index>
        %8 = pop.cast_to_builtin %7 : !pop.scalar<index> to index
        kgen.return %8 : index
    } { name = "g" }
    %1 = kgen.call_signature %0(%idx4) : (index) capturing -> index
    kgen.return %1 : index
}

kgen.func @main_capturing_region() -> index {
    %idx3 = index.constant 3
    %0 = kgen.call @capturing_region(%idx3) : (index) -> index
    kgen.return %0 : index
}

kgen.func @take_closure_no_args(%arg0: !kgen.signature<() capturing -> index>) -> index {
    %0 = kgen.call_signature  %arg0() : () capturing -> index
    kgen.return %0 : index
}

kgen.func @closure_arg() -> index {
    %idx4 = index.constant 98
    %0 = kgen.stage_closure = () capturing -> index {
      %4 = pop.cast_from_builtin %idx4 : index to !pop.scalar<index>
      %5 = kgen.param.constant = <3>
      %6 = pop.cast_from_builtin %5 : index to !pop.scalar<index>
      %7 = pop.add %4, %6 : !pop.scalar<index>
      %8 = pop.cast_to_builtin %7 : !pop.scalar<index> to index
      kgen.return %8 : index
    } { name = "g" }
    %1 = kgen.stage_closure = () capturing -> index {
      %4 = pop.cast_from_builtin %idx4 : index to !pop.scalar<index>
      %5 = kgen.param.constant = <3>
      %6 = pop.cast_from_builtin %5 : index to !pop.scalar<index>
      %7 = pop.add %4, %6 : !pop.scalar<index>
      %8 = pop.cast_to_builtin %7 : !pop.scalar<index> to index
      kgen.return %8 : index
    } { name = "h" }
    %2 = kgen.call @take_closure_no_args(%0) : (!kgen.signature<() capturing -> index>) -> index
    kgen.return %2 : index
}

kgen.export @main_capturing_region
kgen.export @closure_arg

// CHECK: --- 'main_capturing_region' returned 7
// CHECK: --- 'closure_arg' returned 101
