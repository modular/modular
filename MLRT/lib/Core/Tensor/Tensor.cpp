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

#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

Tensor Tensor::createOwned(TensorSpec spec, CompactCPUDevicePtr cpuDevice,
                           std::optional<size_t> alignment) {
  return Tensor(
      TensorBufferRef::createOwned(cpuDevice, spec.getSizeInBytes(), alignment),
      std::move(spec));
}

Tensor Tensor::takeOwned(TensorSpec spec, CompactCPUDevicePtr cpuDevice,
                         void *takeFrom) {
  return Tensor(
      TensorBufferRef::take(cpuDevice, spec.getSizeInBytes(), takeFrom),
      std::move(spec));
}

Tensor Tensor::createBorrowed(void *buffer, TensorSpec spec,
                              std::optional<size_t> alignment) {
  return Tensor(TensorBufferRef::createWithNonTrackedMemory(
                    buffer, spec.getSizeInBytes(), alignment),
                std::move(spec));
}

Tensor Tensor::createBorrowed(const void *buffer, TensorSpec spec,
                              std::optional<size_t> alignment) {
  return Tensor(
      TensorBufferRef::createWithNonTrackedMemory(
          const_cast<void *>(buffer), spec.getSizeInBytes(), alignment),
      std::move(spec));
}

Tensor Tensor::borrow() const {
  return Tensor(bufferRef.borrow(), TensorSpec(spec));
}

void Tensor::printDebug(raw_ostream &os) const {
  os << "Tensor(";
  os << spec;
  os << ", ";
  os << bufferRef;
  os << ")";
}

//===----------------------------------------------------------------------===//
// Tensor printing functionality.
//===----------------------------------------------------------------------===//

/// This prints the specified tensor data to stdout.
ErrorOrSuccess Tensor::print(ResultOutputStyle style,
                             unsigned precision) const {
  return print(llvm::outs(), style, precision);
}

/// This prints the specified tensor data to stream.
ErrorOrSuccess Tensor::print(raw_ostream &os, ResultOutputStyle style,
                             unsigned precision) const {
  return printTensor(os, getBuffer(), getSpec(), style, precision);
}
