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

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"

namespace M::MOGG {

static bool opHasExtraOutputShapeArgument(Operation *op) {
  return isa<MO::Static::ReshapeOp, MO::Static::BroadcastToOp>(op);
}

static StringRef moOperationName(Operation *op) {
  if (auto customOp = dyn_cast<MO::CustomOp>(op))
    return customOp.getSymbol();

  return op->getName().getStringRef();
}

static std::optional<KernelDeclOperandVariant>
MOTypeToKernelDeclOperand(Type type) {
  auto iface = dyn_cast<MO::MOKernelOperandTypeInterface>(type);
  if (!iface)
    return std::nullopt;
  switch (iface.getKernelOperandKind()) {
  case MO::KernelOperandKind::Tensor:
    return KernelDeclTensorAdaptor{MOGG::IOSpec::InputTensor};
  case MO::KernelOperandKind::Buffer:
    return KernelDeclTensorAdaptor{MOGG::IOSpec::MutableInputTensor};
  case MO::KernelOperandKind::Scalar:
    return KernelDeclScalarAdaptor{};
  case MO::KernelOperandKind::Opaque:
    return KernelDeclOpaqueAdaptor{iface.getKernelOperandOpaqueName()};
  }
  return std::nullopt;
}

static ErrorOrSuccess
matchConvertedMOTypeToKernelDeclType(const KernelDeclOperandVariant &moType,
                                     const KernelDeclOperandVariant &kernelType,
                                     Type originalMOType,
                                     bool allowMutToImmut) {
  if (moType == kernelType)
    return success();

  // Reading mutable storage as an immutable snapshot is sound (the converse
  // is not); see `allowMutToImmut`.
  if (allowMutToImmut) {
    if (const auto *kernelTensor =
            std::get_if<KernelDeclTensorAdaptor>(&kernelType)) {
      const auto *moTensor = std::get_if<KernelDeclTensorAdaptor>(&moType);
      if (moTensor && moTensor->mut && !kernelTensor->mut)
        return success();
    }
  }

  if (std::holds_alternative<KernelDeclVariadicTensorAdaptor>(kernelType)) {
    const auto *tensorType = std::get_if<KernelDeclTensorAdaptor>(&moType);
    if (!tensorType)
      return Error("MO type is not a tensor type");

    const auto &kernelTensor =
        std::get<KernelDeclVariadicTensorAdaptor>(kernelType);
    if (kernelTensor.mut != tensorType->mut)
      return Error("Tensor type mismatch");

    return success();
  }

  if (std::holds_alternative<KernelDeclScalarAdaptor>(kernelType)) {
    const auto *tensorType = std::get_if<KernelDeclTensorAdaptor>(&moType);
    if (!tensorType)
      return Error("MO type is not a tensor type");

    if (tensorType->mut)
      return Error("Expected an immutable tensor type");

    // Opaque codegen handles (`!mogg.tensor`) carry no rank/device and were
    // validated upstream.
    if (auto shaped = dyn_cast<MO::MOShapedTypeInterface>(originalMOType))
      if (shaped.getRank() != 0)
        return Error("MO tensor type should have a rank of 0");

    if (isa<MO::MOHasDeviceTypeInterface>(originalMOType) &&
        !MO::isOnHost(originalMOType))
      return Error("Scalars should always be on the host CPU");

    return success();
  }

  return Error("Mismatched types");
}

template <bool matchOperands>
static ErrorOr<KernelDeclIndexToOperandAdaptorMapping>
matchOperandsOrResultsKernelDecl(ValueRange values,
                                 const KernelDeclFunctionAdaptor &adaptor,
                                 StringRef opName,
                                 bool hasSameVariadicOperandSize,
                                 bool hasExtraOutputShapeArgument,
                                 bool resultLeavesByValue, bool allowMutToImmut,
                                 KernelArgBlame &blame) {
  // Records the signature position of the kernel argument an error is about.
  auto blameArg = [&](const KernelDeclOperandAdaptor &arg) {
    if (arg.sourceDescriptor)
      blame = arg.sourceDescriptor->position;
  };
  static const char *argumentKind = matchOperands ? "input" : "output";

  // Launch arguments (the `execute` device context, the `build_graph` graph
  // builder) have no MO operand; skipping them by kind lets one mapping serve
  // every function variant of a kernel.
  auto skipLaunchArgument =
      [](const llvm::SmallVector<KernelDeclOperandAdaptor> &operands,
         uint64_t &idx) -> std::optional<KernelDeclOperandAdaptor> {
    while (idx < operands.size()) {
      if (!operands[idx].isLaunchArgument())
        return operands[idx];
      ++idx;
    }
    return std::nullopt;
  };

  auto skipChainType = [](auto range, uint64_t &idx) -> std::optional<Type> {
    while (idx < range.size()) {
      if (!isa<MO::ChainType>(range[idx]))
        return range[idx];
      ++idx;
    }
    return std::nullopt;
  };

  const llvm::SmallVector<KernelDeclOperandAdaptor> &kernelOperands =
      matchOperands ? adaptor.inputArguments : adaptor.outputArguments;

  uint64_t mojoVariadicsCount = 0;
  uint64_t mojoNonVariadicsCount = 0;
  uint64_t moOperandsCount = 0;
  uint64_t moOpIndex = 0, kernelIndex = 0;

  while (true) {
    auto moType = skipChainType(values.getTypes(), moOpIndex);
    if (!moType.has_value())
      break;
    ++moOpIndex;
    ++moOperandsCount;
  }

  while (true) {
    auto kernelType = skipLaunchArgument(kernelOperands, kernelIndex);
    if (!kernelType.has_value())
      break;
    if (kernelType->isVariadicTensorType())
      ++mojoVariadicsCount;
    else
      ++mojoNonVariadicsCount;
    ++kernelIndex;
  }

  uint64_t variadicSize = 0;
  if (mojoVariadicsCount > 0) {
    if (mojoVariadicsCount > 1 && !hasSameVariadicOperandSize) {
      return Error(llvm::formatv(
          "Op '{}' binds {} variadic {} tensor arguments, but an op's flat "
          "operand list carries no group boundaries, so it can bind at most "
          "one per operand list. Several equal-sized groups require a "
          "dedicated op that implements SameVariadicOperandSize",
          opName, mojoVariadicsCount, argumentKind));
    }
    uint64_t totalVariadicOperands =
        moOperandsCount < mojoNonVariadicsCount
            ? 0
            : moOperandsCount - mojoNonVariadicsCount;
    if (totalVariadicOperands % mojoVariadicsCount != 0) {
      return Error(llvm::formatv(
          "Op '{}' has a total {} variadic operands divided "
          "between {} variadic kernel operands. They are expected "
          "to divide evenly",
          opName, totalVariadicOperands, mojoVariadicsCount));
    }
    variadicSize = totalVariadicOperands / mojoVariadicsCount;
  }

  KernelDeclIndexToOperandAdaptorMapping mapping;
  moOpIndex = 0, kernelIndex = 0;
  uint64_t currVariadicSize = 0;
  bool matchingVariadicTensor = false;

  while (true) {
    auto moType = skipChainType(values.getTypes(), moOpIndex);
    auto kernelType = skipLaunchArgument(kernelOperands, kernelIndex);

    if (!moType.has_value() || !kernelType.has_value())
      break;

    auto moTypeToKernelDeclOperand = MOTypeToKernelDeclOperand(*moType);
    if (!moTypeToKernelDeclOperand.has_value()) {
      return Error(
          llvm::formatv("Unsupported op argument type at index {}", moOpIndex));
    }

    auto errorOnMatch = matchConvertedMOTypeToKernelDeclType(
        *moTypeToKernelDeclOperand, kernelType->underlyingType, *moType,
        allowMutToImmut);

    if (errorOnMatch.isError()) {
      blameArg(*kernelType);
      return Error(llvm::formatv(
          "Op '{}' {} at position {} does not match registered kernel operand "
          "at position {}: {} vs {}. Detail: {}",
          opName, argumentKind, moOpIndex,
          kernelType->sourceDescriptor->position, *moTypeToKernelDeclOperand,
          *kernelType, errorOnMatch.getError()));
    }

    mapping[moOpIndex] = *kernelType;

    if (kernelType->isVariadicTensorType()) {
      matchingVariadicTensor = true;
      if (++currVariadicSize == variadicSize) {
        matchingVariadicTensor = false;
        currVariadicSize = 0;
      }
    }

    ++moOpIndex;
    kernelIndex += !matchingVariadicTensor;
  }

  if (matchingVariadicTensor)
    skipLaunchArgument(kernelOperands, ++kernelIndex);

  bool moOpEnd = moOpIndex == values.size();
  bool kernelEnd = kernelIndex == kernelOperands.size();

  if (hasExtraOutputShapeArgument)
    return mapping;

  if (!kernelEnd) {
    blameArg(kernelOperands[kernelIndex]);
    return Error(llvm::formatv(
        "Could not provide all {} arguments to op '{}': missing operands "
        "starting from argument named '{}'",
        argumentKind, opName,
        kernelOperands[kernelIndex].sourceDescriptor->sourceName.str()));
  }

  if constexpr (matchOperands) {
    if (!moOpEnd) {
      return Error(
          llvm::formatv("Op '{}' has too many operands: operands starting "
                        "at position {} has no equivalent in kernel",
                        opName, moOpIndex));
    }
    return mapping;
  }

  auto moType = skipChainType(values.getTypes(), moOpIndex);

  // A by-ref result of unrecognized type (KernelDeclUnsupported) is treated as
  // absent. A by-value result has no entry in `values` to match.
  bool hasValidOutputResult =
      !resultLeavesByValue && adaptor.outputResult.has_value() &&
      !std::holds_alternative<KernelDeclUnsupportedAdaptor>(
          adaptor.outputResult->underlyingType);

  if (!moType.has_value() && !hasValidOutputResult)
    return mapping;

  if (moType.has_value() && !hasValidOutputResult) {
    return Error(llvm::formatv(
        "Op '{}' has too many results: results starting at position {} have no "
        "equivalent output argument or function result in kernel",
        opName, moOpIndex));
  }

  if (!moType.has_value() && hasValidOutputResult) {
    return Error(llvm::formatv(
        "Op '{}' is missing a result for kernel function result of type {}",
        opName, *adaptor.outputResult));
  }

  auto moTypeToKernelDeclOperand = MOTypeToKernelDeclOperand(*moType);
  if (!moTypeToKernelDeclOperand.has_value()) {
    return Error(
        llvm::formatv("Op '{}': Unsupported op result type at position {}",
                      opName, moOpIndex));
  }

  auto errorOnMatch = matchConvertedMOTypeToKernelDeclType(
      *moTypeToKernelDeclOperand, adaptor.outputResult->underlyingType, *moType,
      /*allowMutToImmut=*/false);
  if (errorOnMatch.isError()) {
    blameArg(*adaptor.outputResult);
    return Error(llvm::formatv(
        "Op '{}' output at position {} does not match kernel function result; "
        "op has {} but kernel has {}. Detail: {}",
        opName, moOpIndex, *moTypeToKernelDeclOperand, *adaptor.outputResult,
        errorOnMatch.getError()));
  }
  mapping[moOpIndex] = *adaptor.outputResult;

  return mapping;
}

ErrorOr<KernelDeclOperandMapping> mapToKernelImplementation(
    ValueRange operands, ValueRange results,
    const KernelDeclFunctionAdaptor &adaptor, StringRef opName,
    bool hasSameVariadicOperandSize, bool hasExtraOutputShapeArgument,
    bool resultLeavesByValue, bool allowMutToImmut, KernelArgBlame &blame) {
  auto matchedInputsOrError = matchOperandsOrResultsKernelDecl<true>(
      operands, adaptor, opName, hasSameVariadicOperandSize,
      hasExtraOutputShapeArgument, /*resultLeavesByValue=*/false,
      allowMutToImmut, blame);
  if (matchedInputsOrError.isError())
    return matchedInputsOrError.takeError();

  // Outputs are destination-passing (mutable); the mut->immut widening only
  // applies to read-only inputs.
  auto matchedOutputsOrError = matchOperandsOrResultsKernelDecl<false>(
      results, adaptor, opName, hasSameVariadicOperandSize,
      hasExtraOutputShapeArgument, resultLeavesByValue,
      /*allowMutToImmut=*/false, blame);
  if (matchedOutputsOrError.isError())
    return matchedOutputsOrError.takeError();

  return KernelDeclOperandMapping{matchedInputsOrError.takeValue(),
                                  matchedOutputsOrError.takeValue()};
}

ErrorOr<KernelDeclOperandMapping>
mapToKernelImplementation(ValueRange operands, ValueRange results,
                          const KernelDeclFunctionAdaptor &adaptor,
                          StringRef opName, bool hasSameVariadicOperandSize,
                          bool hasExtraOutputShapeArgument,
                          bool resultLeavesByValue, bool allowMutToImmut) {
  KernelArgBlame unused;
  return mapToKernelImplementation(
      operands, results, adaptor, opName, hasSameVariadicOperandSize,
      hasExtraOutputShapeArgument, resultLeavesByValue, allowMutToImmut,
      unused);
}

ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor,
                              KernelArgBlame &blame) {
  return mapToKernelImplementation(
      op->getOperands(), op->getResults(), adaptor, moOperationName(op),
      isa<MO::SameVariadicOperandSizeInterface>(op),
      opHasExtraOutputShapeArgument(op), /*resultLeavesByValue=*/false,
      /*allowMutToImmut=*/false, blame);
}

ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor) {
  KernelArgBlame unused;
  return mapMOOpToKernelImplementation(op, adaptor, unused);
}

} // namespace M::MOGG
