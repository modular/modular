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

#ifndef GENERICML_API_TENSOR_MAP_H
#define GENERICML_API_TENSOR_MAP_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Core/Tensor/Tensor.h"

namespace M {

/// A `TensorMap` owns the a map from tensor names to async tensor types along
/// with a cpuDevice used for executing a model.
struct TensorMap {

  TensorMap(AsyncRT::CPUDevice &cpuDevice) : cpuDevice{cpuDevice} {}

  TensorMap(llvm::StringMap<AnyAsyncValueRef> tensors,
            AsyncRT::CPUDevice &cpuDevice)
      : tensors{std::move(tensors)}, cpuDevice{cpuDevice} {}

  /// Map of tensor names to  async tensor values useful for representing inputs
  /// or outputs in `M_executeModelSync`.
  llvm::StringMap<AsyncRT::AnyAsyncValueRef> tensors;

  /// The underlying cpuDevice used for execution
  AsyncRT::CPUDevice &cpuDevice;
};

} // namespace M

#endif // GENERICML_API_TENSOR_MAP_H
