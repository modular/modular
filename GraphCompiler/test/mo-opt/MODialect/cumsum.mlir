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

// CHECK: mo.graph @cumsum(%[[ARG0:.*]]: !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32> {
mo.graph @cumsum(%arg0 : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  // CHECK-NEXT: %[[RES:.*]] = mo.cumsum(%[[ARG0]]) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  %res = mo.cumsum(%arg0) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
