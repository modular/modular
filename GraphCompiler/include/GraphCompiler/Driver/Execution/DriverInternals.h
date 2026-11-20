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
//
// This file defines helper functions used within the ExecutionDriver library.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_EXECUTIONDRIVER_DRIVERINTERNALS_H
#define GENERICML_EXECUTIONDRIVER_DRIVERINTERNALS_H

#include "GraphCompiler/Driver/Execution/DriverCLOptions.h"
#include "GraphCompiler/Support/QuantSpec.h"
#include "Support/ForwardDecls.h"

namespace M {

class ITensorGenerator {
public:
  virtual const MLRT::Tensor &getNextTensor() = 0;
  virtual const MLRT::Tensor &getTensor(size_t sampleIdx) const = 0;
  virtual size_t getSampleCount() const = 0;
  virtual bool isCycled() const = 0;
  virtual ~ITensorGenerator() = default;
};

using ITensorGeneratorPtr = std::unique_ptr<ITensorGenerator>;

ErrorOr<ITensorGeneratorPtr>
createTensorGenerator(size_t tensorIdx, InputGenerationKind defaultGen,
                      ArrayRef<InputDataSpec> inputDataSpecs,
                      const QuantTensorSpec &modelSpec,
                      size_t minimumCpuAlignment, CPUDevice &cpuDevice);

/// Creates an owning tensor generator per model input, using inputGenKind
/// unless inputDataSpecs gives a per-tensor InputGenerationKind or file.
ErrorOr<SmallVector<ITensorGeneratorPtr, 3>>
createTensorGenerators(InputGenerationKind inputGenKind,
                       ArrayRef<QuantTensorSpec> inputShapeSpecs,
                       ArrayRef<InputDataSpec> inputDataSpecs,
                       size_t minimumCpuAlignment, CPUDevice &cpuDevice);

} // namespace M

#endif // GENERICML_EXECUTIONDRIVER_DRIVERINTERNALS_H
