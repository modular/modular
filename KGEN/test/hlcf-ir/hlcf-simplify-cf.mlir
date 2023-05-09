// RUN: kgen-opt %s -simplify-cf | FileCheck %s

// CHECK-LABEL: @remove_trivial_loop_0
kgen.func @remove_trivial_loop_0() -> () {
  // CHECK-NOT: hlcf.loop
  // CHECK-NEXT: return
  hlcf.loop {
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_1
kgen.func @remove_trivial_loop_1(%arg0: index) -> index {
  // CHECK-NOT: hlcf.loop
  // CHECK-NEXT: return %arg0
  %r = hlcf.loop () -> index {
    hlcf.break %arg0: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_2
// This loop shouldn't be removed as it has continue.
kgen.func @remove_trivial_loop_2(%cond: i1, %arg0: index) -> index {
  // CHECK: hlcf.loop
  // CHECK: return
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      hlcf.continue
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_3
// This loop shouldn't be removed as it has two breaks.
kgen.func @remove_trivial_loop_3(%cond: i1, %arg0: index, %arg1: index) -> index {
  // CHECK: hlcf.loop
  // CHECK: return
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      hlcf.break %arg1: index
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_4
// This loop can be removed as the return doesn't make the transformation incorrect.
kgen.func @remove_trivial_loop_4(%cond: i1, %arg0: index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.loop
  // CHECK:     return
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      kgen.return %arg1: index
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_5
// Here we can remove the outer loop despite the presence of break and continue
// in the inner loop (which can't be removed).
kgen.func @remove_trivial_loop_5(%cond: i1, %arg0: index, %arg1: index) -> index {
  // CHECK-COUNT-1: hlcf.loop
  // CHECK:         return
  %r = hlcf.loop () -> index {
    %t = hlcf.loop () -> index {
      hlcf.if %cond {
        hlcf.continue
      } else {
        hlcf.yield
      }
      hlcf.break %arg0: index
    }
    hlcf.break %t: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_6
// This loop can be removed even though the break is to the outer loop.
kgen.func @remove_trivial_loop_6(%cond: i1, %arg0: index, %arg1: index) -> index {
  // TODO: We should be able to delete both loops here, but we only manage to
  // delete the inner one now.
  // CHECK-NOT: hlcf.loop {
  // CHECK:     return
  hlcf.loop "outer" {
    hlcf.loop {
      hlcf.break "outer"
    }
    hlcf.break
  }
  kgen.return %arg0: index
}

// CHECK-LABEL: @remove_trivial_loop_7
// This loop can't be removed because the continue is for the outer loop.
kgen.func @remove_trivial_loop_7() {
  // CHECK-COUNT-2: hlcf.loop
  // CHECK:         return
  hlcf.loop "outer" {
    hlcf.loop {
      hlcf.continue "outer"
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_8
// The inner loop can be removed, the outer cannot.
kgen.func @remove_trivial_loop_8(%cond: i1) {
  // CHECK-COUNT-1: hlcf.loop
  // CHECK:         return
  hlcf.loop {
    hlcf.if %cond {
      hlcf.continue
    } else {
      hlcf.loop {
        hlcf.break
      }
      hlcf.yield
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_9
// Both loops can be removed.
kgen.func @remove_trivial_loop_9(%cond: i1) {
  // CHECK-NOT:  hlcf.loop
  // CHECK-NEXT: return
  hlcf.loop {
    hlcf.loop {
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_10
kgen.func @remove_trivial_loop_10(%cond: i1) {
  // CHECK-NOT:  hlcf.loop
  // CHECK-NEXT: return
  hlcf.loop {
    hlcf.loop {
     kgen.return
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_11
// Only the outer loop can be removed.
kgen.func @remove_trivial_loop_11(%cond: i1) {
  // CHECK:      hlcf.loop
  // CHECK-NOT:  hlcf.loop
  // CHECK: return
  hlcf.loop () {
    hlcf.loop () {
      hlcf.if %cond {
        hlcf.continue
      } else {
        kgen.return
      }
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_12
// Only the outer loop can be removed.
kgen.func @remove_trivial_loop_12(%cond: i1) {
  // CHECK:      hlcf.loop
  // CHECK-NOT:  hlcf.loop
  // CHECK-NEXT:   hlcf.if
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   else
  // CHECK-NEXT:     hlcf.break
  // CHECK:        kgen.return
  // CHECK:      kgen.return
  hlcf.loop () {
    hlcf.loop () {
      hlcf.if %cond {
        hlcf.yield
      } else {
        hlcf.break
      }
      kgen.return
    }
    hlcf.break
  }
  kgen.return
}
