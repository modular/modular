//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK:   mo.graph @resize.nearest(%[[ARG0:.*]]: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
mo.graph @resize.nearest(%arg0 : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    //  CHECK-NEXT: %[[ARG1:.*]] = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.nearest(%[[ARG0]], %[[ARG1]]) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>, round_mode = 2] : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    %output = mo.resize.nearest(%arg0, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>, round_mode = 2]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK:   mo.graph @resize.nearest3(%[[ARG0:.*]]: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
mo.graph @resize.nearest3(%arg0 : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    //  CHECK-NEXT: %[[ARG1:.*]] = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.nearest(%[[ARG0]], %[[ARG1]]) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>] : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    %output = mo.resize.nearest(%arg0, %size) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK:   mo.graph @resize_linear(%arg0: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 4, 4], f32> {
mo.graph @resize_linear(%arg0 : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 4, 4], f32> {
    // CHECK-NEXT: %[[ARG1:.*]] = mo.constant <#M.dense_array<1, 1, 4, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %size = mo.constant <#M.dense_array<1, 1, 4, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.linear(%[[ARG0]], %[[ARG1]]) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>, antialias = false] : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 4, 4], f32>
    %output = mo.resize.linear(%arg0, %size) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>, antialias = false]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 4, 4], f32>
    mo.output %output : !mo.tensor<[1, 1, 4, 4], f32>
}

// CHECK:   mo.graph @resize.bicubic(%[[ARG0:.*]]: !mo.tensor<[1, 3, 224, 224], f32>) -> !mo.tensor<[1, 3, 448, 448], f32> {
mo.graph @resize.bicubic(%arg0 : !mo.tensor<[1, 3, 224, 224], f32>) -> !mo.tensor<[1, 3, 448, 448], f32> {
    //  CHECK-NEXT: %[[ARG1:.*]] = mo.constant <#M.dense_array<1, 3, 448, 448> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %size = mo.constant <#M.dense_array<1, 3, 448, 448> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.bicubic(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 3, 448, 448], f32>
    %output = mo.resize.bicubic(%arg0, %size) : (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 3, 448, 448], f32>
    mo.output %output : !mo.tensor<[1, 3, 448, 448], f32>
}
