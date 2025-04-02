// RUN: kgen-opt %s -split-input-file -elaborate-generators -allow-unregistered-dialect | FileCheck %s

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @compare(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp sgt(%arg0, %arg1)
  kgen.return %0 : i1
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> i1 {
  kgen.param.declare value : i1 = <apply(:(index, index) -> i1 @compare, 4294967295, 5)>
  // CHECK-NEXT:  kgen.param.constant: i1 = <1>
  %0 = kgen.param.constant: i1 = <value>
  kgen.return %0 : i1
}
}

// -----

// COM: Cmp falls back to folder when target is not specified

kgen.generator @compare(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp sgt(%arg0, %arg1)
  kgen.return %0 : i1
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> i1 {
  kgen.param.declare value : i1 = <apply(:(index, index) -> i1 @compare, 4294967294, 5)>
  // CHECK-NEXT:  kgen.param.constant: i1 = <1>
  %0 = kgen.param.constant: i1 = <value>
  kgen.return %0 : i1
}

// -----

// COM: Subtraction

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @sub(%arg0: index, %arg1: index) -> index {
  %0 = index.sub %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @sub, 4294967295, 5)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant = <4294967290>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: Shift left

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @shl(%arg0: index, %arg1: index) -> index {
  %0 = index.shl %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @shl, 1, 63)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-9223372036854775808>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: Logical shift right

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.shru %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -1, 4)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <1152921504606846975>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: Arithmetic shift right

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.shrs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -1, 4)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-1>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: And

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.and %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 15, 5)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <5>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned ceil div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.ceildivu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <5>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned ceil div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.ceildivs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed ceil div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.ceildivs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.divu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.divs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned max

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.maxu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <32>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed max

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.maxs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <7>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned min

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.minu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <7>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed min

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.mins %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-32>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: multiplication

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.mul %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <224>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: or

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.or %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <39>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned rem

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.rems %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed rem

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.rems %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}
