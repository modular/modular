// RUN: mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// An index reader that adds its element index, read through a slice, must see
// the index in its own output, not in the slice.

// CHECK-LABEL: Running 'index_reader_through_slice':
// CHECK: 'index_reader_through_slice' returned tensor<4xsi32> [14, 25, 36, 47]
mo.graph @index_reader_through_slice() -> (!mo.tensor<[4], si32>) {
  %x = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0, 0, 0> : tensor<8xsi32>> : !mo.tensor<[8], si32>
  %y = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<8> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %s = mo.slice(%z, %start, %stop, %step) : (
    !mo.tensor<[8], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[4], si32>
  %r = mo.add(%s, %y) : !mo.tensor<[4], si32>
  mo.output %r : !mo.tensor<[4], si32>
}

// The same when the index reader reads its own input through a slice.

// CHECK-LABEL: Running 'index_reader_over_view_through_slice':
// CHECK: 'index_reader_over_view_through_slice' returned tensor<4xsi32> [14, 25, 36, 47]
mo.graph @index_reader_over_view_through_slice() -> (!mo.tensor<[4], si32>) {
  %x = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0> : tensor<16xsi32>> : !mo.tensor<[16], si32>
  %y = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start0 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %start4 = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<8> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %h = mo.slice(%x, %start0, %stop, %step) : (
    !mo.tensor<[16], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[8], si32>
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%h) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  %s = mo.slice(%z, %start4, %stop, %step) : (
    !mo.tensor<[8], si32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[4], si32>
  %r = mo.add(%s, %y) : !mo.tensor<[4], si32>
  mo.output %r : !mo.tensor<[4], si32>
}
