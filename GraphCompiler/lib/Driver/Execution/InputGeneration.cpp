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

#include "AsyncRT/ForwardDecls.h"
#include "GraphCompiler/Driver/Execution/DriverCLOptions.h"
#include "GraphCompiler/Driver/Execution/DriverInternals.h"
#include "GraphCompiler/Driver/Execution/ExecutionDriver.h"
#include "GraphCompiler/Support/QuantSpec.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "Support/AlignedAlloc.h"
#include "Support/AssertStream.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/ML/Fill.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <memory>
#include <numeric>
#include <variant>

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

static ErrorOrSuccess fillWithElements(Tensor &tensor,
                                       InputGenerationKind inputGenKind,
                                       const QuantSpec &quantSpec) {
  uint8_t elementBuffer[DType::kMaxElementSizeInBytes];
  auto eltType = tensor.getEltType();

  switch (inputGenKind) {
  case InputGenerationKind::kRandom:
    return fillRandom(tensor.getMutableBuffer(),
                      tensor.getShape().getNumElements(), eltType);

  case InputGenerationKind::kOnes: {
    ErrorOrSuccess result = getScalarOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kNegOnes: {
    ErrorOrSuccess result = getScalarNegativeOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kZeros:
    if (quantSpec.isQuantized()) {
      // FIXME: we need to handle all types, and refactor this into a kernel
      if (tensor.getEltType() != DType::ui8 &&
          tensor.getEltType() != DType::si8)
        return Error("when filling quantized tensors with zeros, only 8 bit "
                     "storage types are supported");
      auto params = quantSpec.getParams();
      if (quantSpec.getKind() == QuantSpec::Kind::uniformPerAxis &&
          params.size() != 1)
        return Error("when filling quantized tensors with zeros, only a single "
                     "scale/zeroPoint is supported");
      memset(elementBuffer, (uint8_t)params[0].zeroPoint,
             sizeof(elementBuffer));
    } else
      memset(elementBuffer, 0, sizeof(elementBuffer));
    break;
  }

  return fillHomogeneous(tensor.getMutableBuffer(),
                         tensor.getShape().getNumElements(), eltType,
                         elementBuffer);
}

static ErrorOr<Tensor> allocateInput(const QuantTensorSpec &modelSpec,
                                     CPUDevice &cpuDevice, size_t alignment) {
  return Tensor::createOwned({modelSpec, modelSpec.getEltType()}, cpuDevice,
                             alignment);
}

static ErrorOrSuccess fillInput(InputGenerationKind inputGenKind,
                                const QuantTensorSpec &modelSpec,
                                Tensor &tensor) {
  if (auto err =
          fillWithElements(tensor, inputGenKind, modelSpec.getQuantSpec()))
    return err.takeError();
  return success();
}

static ErrorOr<Tensor> allocateAndFillInput(InputGenerationKind inputGenKind,
                                            const QuantTensorSpec &modelSpec,
                                            CPUDevice &cpuDevice,
                                            size_t alignment) {
  auto resultOr = allocateInput(modelSpec, cpuDevice, alignment);
  if (resultOr.isError())
    return resultOr.takeError();
  auto result = std::move(*resultOr);
  if (auto err = fillInput(inputGenKind, modelSpec, result))
    return err.takeError();
  return result;
}

// Tensor that has been allocated and filled with an InputGenerationKind.
class FilledTensorGenerator : public ITensorGenerator {
public:
  static ErrorOr<FilledTensorGenerator> create(InputGenerationKind genKind,
                                               const QuantTensorSpec &modelSpec,
                                               CPUDevice &cpuDevice,
                                               size_t alignment) {
    auto tensorOr =
        allocateAndFillInput(genKind, modelSpec, cpuDevice, alignment);
    if (tensorOr.isError())
      return tensorOr.takeError();
    return FilledTensorGenerator(std::move(*tensorOr));
  }

  const Tensor &getNextTensor() final { return tensor; }

  const Tensor &getTensor(size_t sampleIdx) const final { return tensor; }

  size_t getSampleCount() const final { return 1; }

  bool isCycled() const final { return true; }

private:
  FilledTensorGenerator(Tensor &&tensor) : tensor(std::move(tensor)) {}
  Tensor tensor;
};

// Tensors backed by a file loaded into memory on construction;
// getNextTensor() strides through it.
class FileTensorGenerator : public ITensorGenerator {
public:
  static ErrorOr<FileTensorGenerator>
  create(const std::filesystem::path &filename, const QuantTensorSpec &spec,
         CPUDevice &cpuDevice, size_t alignment, TensorGeneratorFlag flags) {
    std::vector<Tensor> tensors;
    std::string errorMsg;
    auto data = mlir::openInputFile(filename.string(), llvm::Align(alignment),
                                    &errorMsg);
    if (!data) {
      return Error("Could not open input file.");
    }
    auto stride = spec.getEltType().getSizeInBytes(spec.getNumElements());
    if (!isSet(flags, TensorGeneratorFlag::kTruncate) &&
        data->getBufferSize() % stride != 0) {
      return Error("--input-data file size not a multiple of tensor size. "
                   "Add Truncate attribute to --input-data flag to "
                   "allow this behaviour.");
    }
    auto dataStart = data->getBufferStart();
    for (size_t i = 0, n = data->getBufferSize() / stride; i < n; ++i) {
      auto tensor =
          Tensor::createOwned({spec, spec.getEltType()}, cpuDevice, alignment);
      auto tensorBuf = static_cast<char *>(tensor.getMutableBuffer());
      std::memcpy(tensorBuf, dataStart + i * stride, tensor.getSizeInBytes());
      tensors.push_back(std::move(tensor));
    }

    if (tensors.empty())
      return Error("input file contains 0 tensors");

    return FileTensorGenerator(std::move(tensors), flags);
  }
  const Tensor &getNextTensor() final {
    ASSERT_STREAM((isSet(flags, TensorGeneratorFlag::kCycle) || !repeated),
                  << "All tensors in FileTensor have already been generated. "
                     "Add Cycle attribute to --input-data to loop over "
                     "available tensors.");
    if (pos + 1 >= tensors.size())
      repeated = true;
    auto curr = pos;
    pos = (pos + 1) % tensors.size();
    return tensors[curr];
  }

  const Tensor &getTensor(size_t sampleIdx) const final {
    if (!isCycled())
      ASSERT_STREAM(sampleIdx < tensors.size(),
                    << "Requested sampleIdx out of range");
    return tensors[sampleIdx % tensors.size()];
  }

  size_t getSampleCount() const final { return tensors.size(); }

  bool isCycled() const final {
    return isSet(flags, TensorGeneratorFlag::kCycle);
  }

private:
  FileTensorGenerator(std::vector<Tensor> &&tensors, TensorGeneratorFlag flags)
      : tensors(std::move(tensors)), flags(flags) {}

  size_t pos{0};
  std::vector<Tensor> tensors; // each tensor holds a reference to a position in
                               // data so this shouldn't be too big
  TensorGeneratorFlag flags;
  bool repeated{false};
};

ErrorOr<ITensorGeneratorPtr>
M::createTensorGenerator(size_t tensorIdx, InputGenerationKind defaultGen,
                         ArrayRef<InputDataSpec> inputDataSpecs,
                         const QuantTensorSpec &modelSpec,
                         size_t minimumCpuAlignment, CPUDevice &cpuDevice) {
  std::optional<InputDataSpec> spec;
  for (const auto &inputDataSpec : inputDataSpecs) {
    if (inputDataSpec.tensorIdx == tensorIdx)
      spec = inputDataSpec;
  }

  // A generation function in the tensor spec overrides the default.
  InputGenerationKind *inputGenKindPtr = nullptr;
  if (spec)
    inputGenKindPtr = std::get_if<InputGenerationKind>(&spec->inputGen);

  if (!spec || inputGenKindPtr) {
    auto filledGeneratorOr = FilledTensorGenerator::create(
        inputGenKindPtr ? *inputGenKindPtr : defaultGen, modelSpec, cpuDevice,
        minimumCpuAlignment);
    if (filledGeneratorOr.isError())
      return filledGeneratorOr.takeError();
    return std::make_unique<FilledTensorGenerator>(
        std::move(*filledGeneratorOr));
  }

  auto filename = *std::get_if<std::filesystem::path>(&spec->inputGen);
  auto fileGeneratorOr = FileTensorGenerator::create(
      filename, modelSpec, cpuDevice, minimumCpuAlignment, spec->genFlags);
  if (fileGeneratorOr.isError())
    return fileGeneratorOr.takeError();
  return std::make_unique<FileTensorGenerator>(std::move(*fileGeneratorOr));
}

ErrorOr<SmallVector<ITensorGeneratorPtr, 3>>
M::createTensorGenerators(InputGenerationKind inputGenKind,
                          ArrayRef<QuantTensorSpec> inputShapeSpecs,
                          ArrayRef<InputDataSpec> inputDataSpecs,
                          size_t minimumCpuAlignment, CPUDevice &cpuDevice) {
  auto numInputs = inputShapeSpecs.size();

  for (const auto &inputData : inputDataSpecs) {
    if (inputData.tensorIdx >= numInputs) {
      return Error(
          Twine("Tensor index (") + Twine(inputData.tensorIdx) +
          ") passed to --input-data larger than number of inputs in model (" +
          Twine(numInputs) + ").");
    }
  }

  SmallVector<ITensorGeneratorPtr, 3> inputTensorGenerators;
  for (size_t i = 0; i < numInputs; ++i) {
    const auto &modelSpec = inputShapeSpecs[i];
    auto generatorOrError =
        createTensorGenerator(i, inputGenKind, inputDataSpecs, modelSpec,
                              minimumCpuAlignment, cpuDevice);
    if (generatorOrError) {
      return Error("Input " + llvm::Twine(i) + ": " +
                   generatorOrError.getError());
    }
    inputTensorGenerators.push_back(std::move(*generatorOrError));
  }
  return inputTensorGenerators;
}
