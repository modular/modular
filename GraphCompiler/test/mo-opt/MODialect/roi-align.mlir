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

// CHECK:   mo.graph @roi_align(%[[ARG0:.*]]: !mo.tensor<[1, 10, 10, 1], f32>, %[[ROIS:.*]]: !mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5, 5, 1], f32> {
mo.graph @roi_align(%inp: !mo.tensor<[1, 10, 10, 1], f32>, %rois: !mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5, 5, 1], f32> {
    %output_height = mo.constant <#M.dense_array<5> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>

    %output = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[1, 10, 10, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    mo.output %output : !mo.tensor<[1, 5, 5, 1], f32>
}
