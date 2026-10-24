// RUN: mt --execute %s --result-output-style=full | FileCheck %s

// # Equivalent in python:
// import numpy as np
// input = np.arange(120).reshape(2, 3, 4, 5)
// index = np.array([0, 1, 1, 2, 2, 1, 0, 1, 2]).reshape(3, 3)
// out = input[:, :, index, index].flatten()

// CHECK: 'advanced_indexing_getitem' returned tensor<2x3x3x3xsi32>
// CHECK-SAME: [0, 6, 6, 12, 12, 6, 0, 6, 12, 20, 26, 26, 32, 32, 26, 20, 26, 32, 40, 46, 46, 52, 52, 46, 40, 46, 52, 60, 66, 66, 72, 72, 66, 60, 66, 72, 80, 86, 86, 92, 92, 86, 80, 86, 92, 100, 106, 106, 112, 112, 106, 100, 106, 112]
mo.graph @advanced_indexing_getitem() -> !mo.tensor<[2, 3, 3, 3], si32> {
  %start = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %stop = mo.constant <#M.dense_array<120> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %input = mo.range(%start, %stop, %step) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[120], si32>
  %input_reshape = mo.static.reshape(%input): !mo.tensor<[120], si32> -> !mo.tensor<[2, 3, 4, 5], si32>

  %index_tensor = mo.constant <#M.dense_array<0, 1, 1, 2, 2, 1, 0, 1, 2> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>

  %out = mo.custom [symbol = "advanced_indexing_getitem", device = #M.device_ref<"cpu", 0>, parameters = { start_axis = 2 }] (%input_reshape, %index_tensor, %index_tensor) : (
    !mo.tensor<[2, 3, 4, 5], si32>, !mo.tensor<[3, 3], si32>, !mo.tensor<[3, 3], si32>
  ) -> !mo.tensor<[2, 3, 3, 3], si32>
  mo.output %out : !mo.tensor<[2, 3, 3, 3], si32>
}

// # Equivalent in python:
// import numpy as np
// input = np.arange(120).reshape(2, 3, 4, 5)
// index1 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]).reshape(3, 3)
// index2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]).reshape(3, 3)
// updates = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9]).reshape(3, 3)
// input[:, :, index1, index2] = updates
// print(input.flatten())

// CHECK: 'advanced_indexing_setitem' returned tensor<2x3x4x5xsi32>
// CHECK-SAME: [-1, -4, -7, 3, 4, -2, -5, -8, 8, 9, -3, -6, -9, 13, 14, 15, 16, 17, 18, 19, -1, -4, -7, 23, 24, -2, -5, -8, 28, 29, -3, -6, -9, 33, 34, 35, 36, 37, 38, 39, -1, -4, -7, 43, 44, -2, -5, -8, 48, 49, -3, -6, -9, 53, 54, 55, 56, 57, 58, 59, -1, -4, -7, 63, 64, -2, -5, -8, 68, 69, -3, -6, -9, 73, 74, 75, 76, 77, 78, 79, -1, -4, -7, 83, 84, -2, -5, -8, 88, 89, -3, -6, -9, 93, 94, 95, 96, 97, 98, 99, -1, -4, -7, 103, 104, -2, -5, -8, 108, 109, -3, -6, -9, 113, 114, 115, 116, 117, 118, 119]
mo.graph @advanced_indexing_setitem() -> (!mo.buffer<[2, 3, 4, 5], si32>, !mo.chain) {
  %start = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %stop = mo.constant <#M.dense_array<120> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[], si32>

  // Setup your buffer
  %ch0 = mo.chain.create()
  %buffer_input = mo.buffer.create : !mo.buffer<[2, 3, 4, 5], si32>
  %input = mo.range(%start, %stop, %step) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[120], si32>
  %input_reshape = mo.static.reshape(%input): !mo.tensor<[120], si32> -> !mo.tensor<[2, 3, 4, 5], si32>
  %ch1 = mo.mutable.store(%buffer_input, %input_reshape, %ch0): (!mo.buffer<[2, 3, 4, 5], si32>, !mo.tensor<[2, 3, 4, 5], si32>, !mo.chain) -> !mo.chain

  %data = mo.constant <#M.dense_array<-1, -2, -3, -4, -5, -6, -7, -8, -9> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>
  %data_bcast = mo.static.broadcast_to(%data): !mo.tensor<[3, 3], si32> -> !mo.tensor<[2, 3, 3, 3], si32>

  // NOTE: we want to avoid any combination of indices to be duplicate, since this leads to indeterminism
  %index_tensor1 = mo.constant <#M.dense_array<0, 1, 2, 0, 1, 2, 0, 1, 2> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>
  %index_tensor2 = mo.constant <#M.dense_array<0, 0, 0, 1, 1, 1, 2, 2, 2> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>

  %ch2 = mo.custom [symbol = "advanced_indexing_setitem_inplace", device = #M.device_ref<"cpu", 0>, parameters = { start_axis = 2 }] (%buffer_input, %data_bcast, %index_tensor1, %index_tensor2, %ch1) : (
    !mo.buffer<[2, 3, 4, 5], si32>, !mo.tensor<[2, 3, 3, 3], si32>, !mo.tensor<[3, 3], si32>, !mo.tensor<[3, 3], si32>, !mo.chain
  ) -> !mo.chain

  mo.output %buffer_input, %ch2 : !mo.buffer<[2, 3, 4, 5], si32>, !mo.chain
}

// Equivalent to `advanced_indexing_setitem` but forces a copy.
// CHECK: 'advanced_indexing_setitem_copy' returned tensor<2x3x4x5xsi32>
// CHECK-SAME: [-1, -4, -7, 3, 4, -2, -5, -8, 8, 9, -3, -6, -9, 13, 14, 15, 16, 17, 18, 19, -1, -4, -7, 23, 24, -2, -5, -8, 28, 29, -3, -6, -9, 33, 34, 35, 36, 37, 38, 39, -1, -4, -7, 43, 44, -2, -5, -8, 48, 49, -3, -6, -9, 53, 54, 55, 56, 57, 58, 59, -1, -4, -7, 63, 64, -2, -5, -8, 68, 69, -3, -6, -9, 73, 74, 75, 76, 77, 78, 79, -1, -4, -7, 83, 84, -2, -5, -8, 88, 89, -3, -6, -9, 93, 94, 95, 96, 97, 98, 99, -1, -4, -7, 103, 104, -2, -5, -8, 108, 109, -3, -6, -9, 113, 114, 115, 116, 117, 118, 119]
mo.graph @advanced_indexing_setitem_copy() -> !mo.tensor<[2, 3, 4, 5], si32> {
  %start = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %stop = mo.constant <#M.dense_array<120> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[], si32>

  // Setup your buffer
  %input = mo.range(%start, %stop, %step) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[120], si32>
  %input_reshape = mo.static.reshape(%input): !mo.tensor<[120], si32> -> !mo.tensor<[2, 3, 4, 5], si32>

  // NOTE: we want to avoid any combination of indices to be duplicate, since this leads to indeterminism
  %data = mo.constant <#M.dense_array<-1, -2, -3, -4, -5, -6, -7, -8, -9> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>
  %data_bcast = mo.static.broadcast_to(%data): !mo.tensor<[3, 3], si32> -> !mo.tensor<[2, 3, 3, 3], si32>
  %index_tensor1 = mo.constant <#M.dense_array<0, 1, 2, 0, 1, 2, 0, 1, 2> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>
  %index_tensor2 = mo.constant <#M.dense_array<0, 0, 0, 1, 1, 1, 2, 2, 2> : tensor<9xsi32>> : !mo.tensor<[3, 3], si32>

  %out = mo.custom [symbol = "advanced_indexing_setitem", device = #M.device_ref<"cpu", 0>, parameters = { start_axis = 2 }] (%input_reshape, %data_bcast, %index_tensor1, %index_tensor2) : (
    !mo.tensor<[2, 3, 4, 5], si32>, !mo.tensor<[2, 3, 3, 3], si32>, !mo.tensor<[3, 3], si32>, !mo.tensor<[3, 3], si32>
  ) -> !mo.tensor<[2, 3, 4, 5], si32>

  mo.output %out: !mo.tensor<[2, 3, 4, 5], si32>
}
