//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

kgen.func @array_index(%arr: !pop.array<4, i32>) -> i32 {
  %0 = pop.array.get %arr[2] : !pop.array<4, i32>
  kgen.return %0 : i32
}

kgen.export @array_index
