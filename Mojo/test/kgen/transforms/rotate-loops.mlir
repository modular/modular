// RUN: kgen-opt %s -rotate-loops -split-input-file | FileCheck %s

// A exiting test: the store runs on the iteration that breaks too. Rotation
// peels a copy ahead of the loop and leaves one at the end of the body, so the
// test leads and the store still runs once per test.
// CHECK-LABEL: kgen.func @store_before_test(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @store_before_test(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// A value the body behind the test reads is carried from one iteration to the
// next: the copy ahead of the loop supplies the first one's.
// CHECK-LABEL: kgen.func @carries_value_across_test(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   %0 = pop.load    %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index, %arg3 = %0 : index) {
// CHECK-NEXT:     %1 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.store   %arg3, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %3 = index.sub %arg2, %idx1
// CHECK-NEXT:     %4 = pop.load    %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue %3, %4 : index, index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @carries_value_across_test(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %0 = pop.load %p : !kgen.pointer<index>
    %1 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.store %0, %p : !kgen.pointer<index>
    %2 = index.sub %arg0, %idx1
    hlcf.continue %2 : index
  }
  hlcf.return
}

// -----

// A top test has nothing ahead of it to move.
// CHECK-LABEL: kgen.func @top_test_unchanged(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.store   %arg2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @top_test_unchanged(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.store %arg0, %p : !kgen.pointer<index>
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// A pure op ahead of the test is dead on the iteration that breaks, so it is
// left where it is rather than duplicated.
// CHECK-LABEL: kgen.func @pure_op_before_test(%arg0: index) {
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg1 = %arg0 : index) {
// CHECK-NEXT:     %0 = index.mul %arg1, %idx1
// CHECK-NEXT:     %1 = index.cmp eq(%arg1, %idx1)
// CHECK-NEXT:     %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     kgen.call @foo(%0) : (index) -> ()
// CHECK-NEXT:     %3 = index.sub %arg1, %idx1
// CHECK-NEXT:     hlcf.continue %3 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @pure_op_before_test(%n: index) {
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %0 = index.mul %arg0, %idx1
    %1 = index.cmp eq(%arg0, %idx1)
    %c0 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    kgen.call @foo(%0) : (index) -> ()
    %2 = index.sub %arg0, %idx1
    hlcf.continue %2 : index
  }
  hlcf.return
}

// -----

// Effects on both sides of the test rotate as well: duplicating does not
// change how often either one runs.
// CHECK-LABEL: kgen.func @effects_both_sides(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.store   %idx1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @effects_both_sides(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.store %idx1, %p : !kgen.pointer<index>
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// Two static copies of a convergent op no longer bring the same threads
// together, so a convergent function does not rotate.
// CHECK-LABEL: kgen.func @convergent_not_rotated(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-NEXT:     pop.store   %arg2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: } <convergent>
kgen.func @convergent_not_rotated(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
} <{convergent}>

// -----

// A region ahead of the test is duplicated when its jumps stay inside it.
// CHECK-LABEL: kgen.func @region_before_test(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.if %arg2 {
// CHECK-NEXT:     pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.yield
// CHECK-NEXT:   } else {
// CHECK-NEXT:     hlcf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.loop (%arg3 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg3, %idx1
// CHECK-NEXT:     hlcf.if %arg2 {
// CHECK-NEXT:       pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @region_before_test(%p: !kgen.pointer<index>, %n: index,
                                          %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    hlcf.if %c {
      pop.store %arg0, %p : !kgen.pointer<index>
      hlcf.yield
    } else {
      hlcf.yield
    }
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// A stack allocation names one object rather than a value to recompute, and
// its lifetime markers require an operand the allocation defines.
// CHECK-LABEL: kgen.func @stack_allocation_not_rotated(%arg0: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg1 = %arg0 : index) {
// CHECK-NEXT:     %0 = pop.stack_allocation 1 x scalar<index> align 8 marked
// CHECK-NEXT:     pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<scalar<index>>
// CHECK-NEXT:     %1 = index.cmp eq(%arg1, %idx0)
// CHECK-NEXT:     %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<scalar<index>>
// CHECK-NEXT:     %3 = index.sub %arg1, %idx1
// CHECK-NEXT:     hlcf.continue %3 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @stack_allocation_not_rotated(%n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %0 = pop.stack_allocation 1 x scalar<index> align 8 marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<scalar<index>>
    %1 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<scalar<index>>
    %2 = index.sub %arg0, %idx1
    hlcf.continue %2 : index
  }
  hlcf.return
}

// -----

// A second exit leaves the body on a path rotation does not duplicate.
// CHECK-LABEL: kgen.func @two_exits_not_rotated(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg3 = %arg1 : index) {
// CHECK-NEXT:     pop.store   %arg3, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.if %arg2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %0 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg3, %idx1
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @two_exits_not_rotated(%p: !kgen.pointer<index>, %n: index,
                                 %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    hlcf.if %c {
      hlcf.break
    } else {
      hlcf.yield
    }
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// Rotation duplicates what it moves, so it gives up past a size limit.
// CHECK-LABEL: kgen.func @too_many_ops_not_rotated(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-COUNT-17: pop.store   %arg2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @too_many_ops_not_rotated(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// A debug record ahead of the test is left in place rather than duplicated, so
// it must read the carried copy once the op it described has moved.
#subprogram = #debuginfo.subprogram<sourceName = <"sp">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#var_v = #debuginfo.local_variable<scope = #subprogram, name = "v"> : !debuginfo.unresolved<index>
// CHECK-LABEL: kgen.func @debug_record_reads_carried_value(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   %0 = pop.load    %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index, %arg3 = %0 : index) {
// CHECK-NEXT:     debuginfo.value #local_variable = %arg3 : index
// CHECK-NEXT:     %1 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.store   %arg3, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %3 = index.sub %arg2, %idx1
// CHECK-NEXT:     %4 = pop.load    %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue %3, %4 : index, index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @debug_record_reads_carried_value(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %0 = pop.load %p : !kgen.pointer<index>
    debuginfo.value #var_v = %0 : index
    %1 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.store %0, %p : !kgen.pointer<index>
    %2 = index.sub %arg0, %idx1
    hlcf.continue %2 : index
  }
  hlcf.return
}

// -----

// A pure consumer of a moving op stays ahead of the test, so what it reads has
// to be carried even though nothing behind the test reads it.
// CHECK-LABEL: kgen.func @pure_consumer_before_test(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   %0 = pop.load    %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index, %arg3 = %0 : index) {
// CHECK-NEXT:     %1 = index.add %arg3, %idx1
// CHECK-NEXT:     %2 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %3 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %3 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.store   %1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %4 = index.sub %arg2, %idx1
// CHECK-NEXT:     %5 = pop.load    %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue %4, %5 : index, index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @pure_consumer_before_test(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %0 = pop.load %p : !kgen.pointer<index>
    %1 = index.add %0, %idx1
    %2 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.store %1, %p : !kgen.pointer<index>
    %3 = index.sub %arg0, %idx1
    hlcf.continue %3 : index
  }
  hlcf.return
}

// -----

// The walk for the loop's exit descends into nested loops, so it has to tell
// whose jump it found: these belong to the inner loop, and the outer one
// rotates.
// CHECK-LABEL: kgen.func @inner_loop_jumps_are_its_own(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop "outer" (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break "outer"
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.loop "inner" (%arg3 = %arg2 : index) {
// CHECK-NEXT:       %3 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:       %4 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:       hlcf.if %4 {
// CHECK-NEXT:         hlcf.break "inner"
// CHECK-NEXT:       } else {
// CHECK-NEXT:         hlcf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %5 = index.sub %arg3, %idx1
// CHECK-NEXT:       hlcf.continue "inner" %5 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue "outer" %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @inner_loop_jumps_are_its_own(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop "outer" (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break "outer"
    } else {
      hlcf.yield
    }
    hlcf.loop "inner" (%arg1 = %arg0 : index) {
      %2 = index.cmp eq(%arg1, %idx0)
      %c1 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
      hlcf.if %c1 {
        hlcf.break "inner"
      } else {
        hlcf.yield
      }
      %3 = index.sub %arg1, %idx1
      hlcf.continue "inner" %3 : index
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue "outer" %1 : index
  }
  hlcf.return
}

// -----

// A labelled break leaves the outer loop from inside the inner one. Pruning
// the walk at the nested loop would miss it and rotate a loop with two exits.
// CHECK-LABEL: kgen.func @inner_loop_breaks_outer(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop "outer" (%arg3 = %arg1 : index) {
// CHECK-NEXT:     pop.store   %arg3, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %0 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break "outer"
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.loop "inner" (%arg4 = %arg3 : index) {
// CHECK-NEXT:       hlcf.if %arg2 {
// CHECK-NEXT:         hlcf.break "outer"
// CHECK-NEXT:       } else {
// CHECK-NEXT:         hlcf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = index.sub %arg4, %idx1
// CHECK-NEXT:       hlcf.continue "inner" %3 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg3, %idx1
// CHECK-NEXT:     hlcf.continue "outer" %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @inner_loop_breaks_outer(%p: !kgen.pointer<index>, %n: index,
                                   %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop "outer" (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break "outer"
    } else {
      hlcf.yield
    }
    hlcf.loop "inner" (%arg1 = %arg0 : index) {
      hlcf.if %c {
        hlcf.break "outer"
      } else {
        hlcf.yield
      }
      %3 = index.sub %arg1, %idx1
      hlcf.continue "inner" %3 : index
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue "outer" %1 : index
  }
  hlcf.return
}

// -----

// A nested loop ahead of the test is duplicated like anything else, since its
// jumps stay inside it.
// CHECK-LABEL: kgen.func @inner_loop_before_test(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop "inner" (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break "inner"
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue "inner" %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.loop "outer" (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break "outer"
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.loop "inner" (%arg3 = %2 : index) {
// CHECK-NEXT:       %3 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:       %4 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:       hlcf.if %4 {
// CHECK-NEXT:         hlcf.break "inner"
// CHECK-NEXT:       } else {
// CHECK-NEXT:         hlcf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %5 = index.sub %arg3, %idx1
// CHECK-NEXT:       pop.store   %5, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:       hlcf.continue "inner" %5 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.continue "outer" %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @inner_loop_before_test(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop "outer" (%arg0 = %n : index) {
    hlcf.loop "inner" (%arg1 = %arg0 : index) {
      pop.store %arg1, %p : !kgen.pointer<index>
      %2 = index.cmp eq(%arg1, %idx0)
      %c1 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
      hlcf.if %c1 {
        hlcf.break "inner"
      } else {
        hlcf.yield
      }
      %3 = index.sub %arg1, %idx1
      hlcf.continue "inner" %3 : index
    }
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break "outer"
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue "outer" %1 : index
  }
  hlcf.return
}

// -----

// The copy ahead of the loop has no loop to leave, so a nested region whose
// jump reaches out of it is still not duplicated.
// CHECK-LABEL: kgen.func @inner_loop_before_test_breaks_outer(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop "outer" (%arg3 = %arg1 : index) {
// CHECK-NEXT:     hlcf.loop "inner" (%arg4 = %arg3 : index) {
// CHECK-NEXT:       pop.store   %arg4, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:       hlcf.if %arg2 {
// CHECK-NEXT:         hlcf.break "outer"
// CHECK-NEXT:       } else {
// CHECK-NEXT:         hlcf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = index.sub %arg4, %idx1
// CHECK-NEXT:       hlcf.continue "inner" %3 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %0 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break "outer"
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg3, %idx1
// CHECK-NEXT:     hlcf.continue "outer" %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @inner_loop_before_test_breaks_outer(%p: !kgen.pointer<index>,
                                               %n: index,
                                               %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop "outer" (%arg0 = %n : index) {
    hlcf.loop "inner" (%arg1 = %arg0 : index) {
      pop.store %arg1, %p : !kgen.pointer<index>
      hlcf.if %c {
        hlcf.break "outer"
      } else {
        hlcf.yield
      }
      %3 = index.sub %arg1, %idx1
      hlcf.continue "inner" %3 : index
    }
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break "outer"
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue "outer" %1 : index
  }
  hlcf.return
}

// -----

// The region ahead of the test reads another op ahead of it. Both are copied,
// or the copy ahead of the loop would name a value the loop defines.
// CHECK-LABEL: kgen.func @region_reads_another_leading_op(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   %simd = kgen.param.constant: scalar<index> = <1>
// CHECK-NEXT:   %0 = pop.offset %arg0[%simd] : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.if %arg2 {
// CHECK-NEXT:     %1 = pop.load    %0 : !kgen.pointer<index>
// CHECK-NEXT:     pop.store   %1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.yield
// CHECK-NEXT:   } else {
// CHECK-NEXT:     hlcf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.loop (%arg3 = %arg1 : index) {
// CHECK-NEXT:     %1 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:     %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = index.sub %arg3, %idx1
// CHECK-NEXT:     %4 = pop.offset %arg0[%simd] : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.if %arg2 {
// CHECK-NEXT:       %5 = pop.load    %4 : !kgen.pointer<index>
// CHECK-NEXT:       pop.store   %5, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.continue %3 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @region_reads_another_leading_op(%p: !kgen.pointer<index>, %n: index,
                                           %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  %simd = kgen.param.constant: scalar<index> = <1>

  hlcf.loop (%arg0 = %n : index) {
    %0 = pop.offset %p[%simd] : !kgen.pointer<index>
    hlcf.if %c {
      %4 = pop.load %0 : !kgen.pointer<index>
      pop.store %4, %p : !kgen.pointer<index>
      hlcf.yield
    } else {
      hlcf.yield
    }
    %1 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %2 = index.sub %arg0, %idx1
    hlcf.continue %2 : index
  }
  hlcf.return
}

// -----

// The jump leaves a loop outside the one being rotated, so this loop still has
// a single exit and only the duplication rule refuses it.
// CHECK-LABEL: kgen.func @leading_region_jumps_to_outer_loop(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop "outer" (%arg3 = %arg1 : index) {
// CHECK-NEXT:     hlcf.loop "middle" (%arg4 = %arg3 : index) {
// CHECK-NEXT:       hlcf.if %arg2 {
// CHECK-NEXT:         hlcf.break "outer"
// CHECK-NEXT:       } else {
// CHECK-NEXT:         hlcf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       pop.store   %arg4, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:       %1 = index.cmp eq(%arg4, %idx0)
// CHECK-NEXT:       %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:       hlcf.if %2 {
// CHECK-NEXT:         hlcf.break "middle"
// CHECK-NEXT:       } else {
// CHECK-NEXT:         hlcf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = index.sub %arg4, %idx1
// CHECK-NEXT:       hlcf.continue "middle" %3 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %0 = index.sub %arg3, %idx1
// CHECK-NEXT:     hlcf.continue "outer" %0 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @leading_region_jumps_to_outer_loop(%p: !kgen.pointer<index>,
                                              %n: index,
                                              %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop "outer" (%arg0 = %n : index) {
    hlcf.loop "middle" (%arg1 = %arg0 : index) {
      hlcf.if %c {
        hlcf.break "outer"
      } else {
        hlcf.yield
      }
      pop.store %arg1, %p : !kgen.pointer<index>
      %0 = index.cmp eq(%arg1, %idx0)
      %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
      hlcf.if %c0 {
        hlcf.break "middle"
      } else {
        hlcf.yield
      }
      %1 = index.sub %arg1, %idx1
      hlcf.continue "middle" %1 : index
    }
    %2 = index.sub %arg0, %idx1
    hlcf.continue "outer" %2 : index
  }
  hlcf.return
}

// -----

// A region ahead of the test yields a value the body behind it reads, so the
// region's result is what becomes loop-carried.
// CHECK-LABEL: kgen.func @region_result_carried(%arg0: !kgen.pointer<index>, %arg1: index, %arg2: !kgen.scalar<bool>) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   %0 = hlcf.if %arg2 -> index {
// CHECK-NEXT:     pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.yield %arg1 : index
// CHECK-NEXT:   } else {
// CHECK-NEXT:     hlcf.yield %idx1 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.loop (%arg3 = %arg1 : index, %arg4 = %0 : index) {
// CHECK-NEXT:     %1 = index.cmp eq(%arg3, %idx0)
// CHECK-NEXT:     %2 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %2 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     pop.store   %arg4, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %3 = index.sub %arg3, %idx1
// CHECK-NEXT:     %4 = hlcf.if %arg2 -> index {
// CHECK-NEXT:       pop.store   %3, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:       hlcf.yield %3 : index
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield %idx1 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     hlcf.continue %3, %4 : index, index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @region_result_carried(%p: !kgen.pointer<index>, %n: index,
                                 %c: !kgen.scalar<bool>) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    %r = hlcf.if %c -> index {
      pop.store %arg0, %p : !kgen.pointer<index>
      hlcf.yield %arg0 : index
    } else {
      hlcf.yield %idx1 : index
    }
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    pop.store %r, %p : !kgen.pointer<index>
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}
