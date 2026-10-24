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

// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @resize_nearest(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant <#M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
    // expected-error@+1 {{'mo.resize.nearest' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[4], f32>'}}
    %output = mo.resize.nearest(%input, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>, round_mode = 2]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// -----


mo.graph @resize_linear(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant <#M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
    // expected-error@+1 {{'mo.resize.linear' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[4], f32>'}}
    %output = mo.resize.linear(%input, %size) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>, antialias = true] : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}
