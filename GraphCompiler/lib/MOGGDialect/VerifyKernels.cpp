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

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"

using namespace M;
using namespace KGEN;
using namespace M::MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_VERIFYKERNELS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

template <typename Func, typename... Args>
[[nodiscard]] LogicalResult setFlagIfFailed(bool &flag, Func &&func,
                                            Args &&...args) {
  LogicalResult ret = func(std::forward<Args>(args)...);
  if (ret.failed())
    flag |= true;

  return ret;
}

/// Emits an error at the Mojo location of `arg` of `decl`'s `kind` function. A
/// shape function lives in its own decl, so its errors point at the struct.
static mlir::InFlightDiagnostic
emitMojoArgumentError(KernelDeclOp decl, KernelFunctionKind kind,
                      const KernelDeclOperandAdaptor &arg) {
  if (!arg.sourceDescriptor)
    return mlir::emitError(decl.getMojoFunctionLocation(kind));
  return mlir::emitError(
      decl.getMojoArgLocation(kind, arg.sourceDescriptor->position));
}

/// Whether `arg` is declared as an output tensor, wherever it sits.
static bool isDeclaredOutput(const KernelDeclOperandAdaptor &arg) {
  const auto *tensor =
      std::get_if<KernelDeclTensorAdaptor>(&arg.underlyingType);
  if (!tensor)
    return false;
  if (tensor->traitContract)
    return tensor->traitContract->role == TensorArgRole::output;
  return isOutputIOSpec(tensor->ioSpec);
}

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor-based verification (mogg.kernel.decl path)
//===----------------------------------------------------------------------===//

LogicalResult declVerifyShapeFunctionReturnsShape(
    KernelDeclOp decl,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  if (!shapeFunction.has_value())
    return success();
  if (!shapeFunction->outputResult.has_value()) {
    decl->emitError() << "Shape functions must return a shape";
    return failure();
  }
  return success();
}

// Return `s` with its first ASCII letter uppercased, for error messages that
// start with a kernel function name like "execute".
static std::string capitalize(llvm::StringRef s) {
  if (s.empty())
    return "";
  std::string out = s.str();
  out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  return out;
}

/// Walk the input argument lists of `lhs` and `rhs` in parallel, skipping
/// launch args (DeviceContext / DeviceGraphBuilder) from either side, and
/// require pairwise compatibility. `lhsKind` / `rhsKind` name the functions in
/// error messages.
///
/// A pair is compatible when types match exactly, or when the lhs side has a
/// `MutableInputTensor` and the rhs side has a non-mutable tensor (shape /
/// build_graph cannot mutate their inputs, so a relaxed match is allowed).
/// Adaptor equality ignores the `stable` flag on purpose, so a build_graph
/// `Stable` argument aligns with the plain shape-function argument at the same
/// position; `declVerifyStableTensorArguments` owns the rules for where stable
/// arguments may appear.
LogicalResult verifyInputArgumentsAlign(KernelDeclOp decl,
                                        const KernelDeclFunctionAdaptor &lhs,
                                        KernelFunctionKind lhsKind,
                                        const KernelDeclFunctionAdaptor &rhs,
                                        KernelFunctionKind rhsKind) {
  llvm::StringRef lhsName = stringifyKernelFunctionKind(lhsKind);
  llvm::StringRef rhsName = stringifyKernelFunctionKind(rhsKind);
  auto skipLaunchArg =
      [](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
         uint64_t &idx) -> std::optional<KernelDeclOperandAdaptor> {
    while (idx < inputs.size()) {
      const auto &arg = inputs[idx];
      if (!arg.isLaunchArgument())
        return arg;
      ++idx;
    }
    return std::nullopt;
  };

  uint64_t lhsIndex = 0, rhsIndex = 0;
  while (true) {
    const auto lhsArgOpt = skipLaunchArg(lhs.inputArguments, lhsIndex);
    const auto rhsArgOpt = skipLaunchArg(rhs.inputArguments, rhsIndex);
    if (!lhsArgOpt.has_value() || !rhsArgOpt.has_value())
      break;

    auto isCompatibleArgs = [&]() {
      if (*lhsArgOpt == *rhsArgOpt)
        return true;
      return lhsArgOpt->isMutableTensorType() &&
             (rhsArgOpt->isTensorType() && !rhsArgOpt->isMutableTensorType());
    };

    if (!isCompatibleArgs()) {
      return emitMojoArgumentError(decl, lhsKind, *lhsArgOpt)
             << capitalize(rhsName) << " function and " << lhsName
             << " function input argument types do not match. "
             << capitalize(lhsName) << " function input at position "
             << lhsArgOpt->sourceDescriptor->position << " is a "
             << lhsArgOpt->underlyingType << " but corresponding " << rhsName
             << " function argument at position " << rhsIndex << " is a "
             << rhsArgOpt->underlyingType;
    }
    ++lhsIndex;
    ++rhsIndex;
  }

  bool lhsEnd = lhsIndex == lhs.inputArguments.size();
  bool rhsEnd = rhsIndex == rhs.inputArguments.size();

  if (!lhsEnd || !rhsEnd) {
    std::stringstream errorMessage;
    errorMessage << capitalize(lhsName) << " and " << rhsName.str()
                 << " functions do not have the same input tensors. ";
    auto appendUnmatchedInputs =
        [&](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
            uint64_t idx) {
          while (skipLaunchArg(inputs, idx).has_value()) {
            errorMessage << inputs[idx].sourceDescriptor->position << ",";
            ++idx;
          }
        };
    // Point at the first input the other function lacks.
    const KernelDeclFunctionAdaptor &extra = lhsEnd ? rhs : lhs;
    KernelFunctionKind extraKind = lhsEnd ? rhsKind : lhsKind;
    uint64_t extraIndex = lhsEnd ? rhsIndex : lhsIndex;
    if (lhsEnd) {
      errorMessage << capitalize(rhsName) << " function inputs at positions (";
      appendUnmatchedInputs(rhs.inputArguments, rhsIndex);
      errorMessage << ") do not appear in the " << lhsName.str() << " function";
    } else {
      errorMessage << capitalize(lhsName) << " function inputs at positions (";
      appendUnmatchedInputs(lhs.inputArguments, lhsIndex);
      errorMessage << ") do not appear in the " << rhsName.str() << " function";
    }
    std::optional<KernelDeclOperandAdaptor> firstExtra =
        skipLaunchArg(extra.inputArguments, extraIndex);
    if (!firstExtra)
      return decl->emitError(errorMessage.str());
    return emitMojoArgumentError(decl, extraKind, *firstExtra)
           << errorMessage.str();
  }

  return success();
}

LogicalResult declVerifyShapeFunctionInputsMatchExecuteFunction(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  if (!shapeFunction.has_value())
    return success();

  if (adaptor.executeFunction.has_value() &&
      verifyInputArgumentsAlign(decl, *adaptor.executeFunction,
                                KernelFunctionKind::Execute, *shapeFunction,
                                KernelFunctionKind::Shape)
          .failed())
    return failure();

  if (adaptor.buildGraphFunction.has_value() &&
      verifyInputArgumentsAlign(decl, *adaptor.buildGraphFunction,
                                KernelFunctionKind::BuildGraph, *shapeFunction,
                                KernelFunctionKind::Shape)
          .failed())
    return failure();

  return success();
}

/// `build_graph` takes the same arguments as `execute`: a recording stub hands
/// both entry points the same values. The signatures are compared position by
/// position, with two exceptions:
///
/// - The `DeviceContext` of `execute` is a `DeviceGraphBuilder` on
///   `build_graph`.
///
/// - And a `build_graph` tensor may add `Stable` to the `execute` tensor at the
///   same position, as long as that tensor is not fused: a slot publishes the
///   address of a tensor in memory, and a fused input is a compute lambda with
///   no address.
LogicalResult
declVerifyExecuteMatchesBuildGraph(KernelDeclOp decl,
                                   const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value() ||
      !adaptor.buildGraphFunction.has_value())
    return success();
  StringRef executeName =
      stringifyKernelFunctionKind(KernelFunctionKind::Execute);
  StringRef buildGraphName =
      stringifyKernelFunctionKind(KernelFunctionKind::BuildGraph);
  std::string contract =
      (buildGraphName + " takes the same arguments as " + executeName +
       ", with the DeviceContext replaced by a DeviceGraphBuilder")
          .str();
  auto positionOf = [](const KernelDeclOperandAdaptor &arg) -> uint64_t {
    return arg.sourceDescriptor ? arg.sourceDescriptor->position : 0;
  };
  // A trait argument's IOSpec names only its role, so it prints its whole
  // contract: two trait arguments can differ only in fusion or access form.
  auto describe = [&](const KernelDeclOperandAdaptor &arg) -> std::string {
    std::string out;
    llvm::raw_string_ostream os(out);
    if (std::optional<TensorArgContract> contract = arg.getTraitContract())
      os << (arg.isVariadicTensorType() ? "variadic " : "") << "trait "
         << tensorArgContractToAttr(decl.getContext(), *contract);
    else if (std::optional<IOSpec> ioSpec = arg.getIoSpec())
      os << (arg.isVariadicTensorType() ? "variadic " : "")
         << toString(*ioSpec);
    else
      os << arg.underlyingType;
    return out;
  };
  // Tensor equality ignores the `stable` bit, in the trait contract too;
  // everything else must agree. `declVerifyStableTensorArguments` keeps
  // `Stable` off `execute`.
  auto contractIgnoringStable = [](const KernelDeclOperandAdaptor &arg) {
    std::optional<TensorArgContract> contract = arg.getTraitContract();
    if (contract)
      contract->stable = false;
    return contract;
  };
  auto sameArgument = [&](const KernelDeclOperandAdaptor &executeArg,
                          const KernelDeclOperandAdaptor &buildGraphArg) {
    if (!executeArg.isTensorType())
      return executeArg == buildGraphArg;
    return executeArg.isVariadicTensorType() ==
               buildGraphArg.isVariadicTensorType() &&
           executeArg.getIoSpec() == buildGraphArg.getIoSpec() &&
           contractIgnoringStable(executeArg) ==
               contractIgnoringStable(buildGraphArg);
  };

  // `execute` is the reference, so errors point at the `build_graph` side.
  auto compare =
      [&](StringRef kind, ArrayRef<KernelDeclOperandAdaptor> executeArgs,
          ArrayRef<KernelDeclOperandAdaptor> buildGraphArgs) -> LogicalResult {
    if (executeArgs.size() != buildGraphArgs.size())
      return mlir::emitError(
                 decl.getMojoFunctionLocation(KernelFunctionKind::BuildGraph))
             << executeName << " declares " << executeArgs.size() << " " << kind
             << "s but " << buildGraphName << " declares "
             << buildGraphArgs.size() << "; " << contract;
    for (auto [executeArg, buildGraphArg] :
         llvm::zip(executeArgs, buildGraphArgs)) {
      if (executeArg.isContextType() && buildGraphArg.isBuilderType())
        continue;
      if (buildGraphArg.isStableTensorType() && executeArg.isFusedTensorType())
        return emitMojoArgumentError(decl, KernelFunctionKind::BuildGraph,
                                     buildGraphArg)
               << "the " << kind << " at position " << positionOf(buildGraphArg)
               << " of " << buildGraphName
               << " reads a stable address slot, but the " << kind
               << " at position " << positionOf(executeArg) << " of "
               << executeName
               << " is fused; a slot publishes the address of a tensor in "
                  "memory, and a fused input is a functor with none";
      if (!sameArgument(executeArg, buildGraphArg))
        return emitMojoArgumentError(decl, KernelFunctionKind::BuildGraph,
                                     buildGraphArg)
               << "the " << kind << " at position " << positionOf(executeArg)
               << " is " << describe(executeArg) << " on " << executeName
               << " but " << describe(buildGraphArg) << " on " << buildGraphName
               << "; " << contract;
    }
    return success();
  };

  if (compare("input", adaptor.executeFunction->inputArguments,
              adaptor.buildGraphFunction->inputArguments)
          .failed())
    return failure();
  return compare("output", adaptor.executeFunction->outputArguments,
                 adaptor.buildGraphFunction->outputArguments);
}

/// A `Stable` trait argument is a device-graph recording concept: only
/// `build_graph` operands are routed through a slot, so on any other entry
/// point it would read a data pointer as a slot address. Import already
/// rejects a `Stable` output or one alongside `Fused`, and trait arguments on
/// the elementwise entry points.
LogicalResult
declVerifyStableTensorArguments(KernelDeclOp decl,
                                const KernelDeclAdaptor &adaptor) {
  auto rejectStable = [&](const std::optional<KernelDeclFunctionAdaptor> &func,
                          KernelFunctionKind kind) -> LogicalResult {
    if (!func.has_value())
      return success();
    for (const auto &args : {func->inputArguments, func->outputArguments})
      for (const auto &arg : args)
        if (arg.isStableTensorType())
          return decl->emitError()
                 << "Stable trait arguments are only supported on "
                 << stringifyKernelFunctionKind(KernelFunctionKind::BuildGraph)
                 << " functions (argument at position "
                 << (arg.sourceDescriptor ? arg.sourceDescriptor->position : 0)
                 << " of the " << stringifyKernelFunctionKind(kind)
                 << " function)";
    return success();
  };

  // A declarative kernel's `executeFunction` is its `execute_device`.
  if (failed(rejectStable(adaptor.executeFunction,
                          adaptor.isDeclarative()
                              ? KernelFunctionKind::ExecuteDevice
                              : KernelFunctionKind::Execute)))
    return failure();
  return rejectStable(adaptor.executeConfigurationFunction,
                      KernelFunctionKind::ExecuteConfiguration);
}

/// `build_graph` is the device-graph entry point, and the graph compiler only
/// routes operands through stable address slots for tensor trait arguments,
/// so its tensors must be trait-bound rather than the legacy tensor structs.
LogicalResult
declVerifyBuildGraphTakesTraitArguments(KernelDeclOp decl,
                                        const KernelDeclAdaptor &adaptor) {
  if (!adaptor.buildGraphFunction.has_value())
    return success();
  auto sig = adaptor.buildGraphFunction->attr.getSignature();
  bool ok = true;
  for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
    KGEN::LIT::StructType legacyType =
        getLegacyTensorArgType(sig.getArgument(i));
    if (!legacyType)
      continue;
    mlir::emitError(decl.getMojoArgLocation(KernelFunctionKind::BuildGraph, i))
        << stringifyKernelFunctionKind(KernelFunctionKind::BuildGraph)
        << " functions take tensor trait arguments; argument at position " << i
        << " is a '" << legacyType.getSymbol().getLeafReference().getValue()
        << "'; use a trait bound such as 'Some[Input & Tensor]', or "
           "'TensorTuple' for variadic operands";
    ok = false;
  }
  return success(ok);
}

LogicalResult declVerifyBuildGraphHasBuilder(KernelDeclOp decl,
                                             const KernelDeclAdaptor &adaptor) {
  if (!adaptor.buildGraphFunction.has_value())
    return success();

  bool hasBuilder = llvm::any_of(
      adaptor.buildGraphFunction->inputArguments,
      [](const KernelDeclOperandAdaptor &arg) { return arg.isBuilderType(); });

  if (!hasBuilder) {
    decl->emitError()
        << "build_graph method must accept a DeviceGraphBuilder argument";
    return failure();
  }
  return success();
}

LogicalResult declVerifyExecuteFunctionDoesNotReturnATensor(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor) {
  auto doesNotReturnTensor =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value() || !func->outputResult.has_value())
      return success();
    if (func->outputResult->isTensorType()) {
      decl->emitError()
          << "Value semantics kernels (return a tensor) are not yet supported";
      return failure();
    }
    return success();
  };

  if (doesNotReturnTensor(adaptor.executeFunction).failed() ||
      doesNotReturnTensor(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyOutputsAppearBeforeInputs(KernelDeclOp decl,
                                    const KernelDeclAdaptor &adaptor) {
  auto outputsBeforeInputs =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func,
              KernelFunctionKind kind) -> LogicalResult {
    if (!func.has_value())
      return success();
    auto position = [](const KernelDeclOperandAdaptor &arg) {
      return arg.sourceDescriptor->position;
    };
    // Only leading outputs are classified as outputs, so an output after an
    // input shows up among the inputs; its declared role still marks it.
    const KernelDeclOperandAdaptor *firstInput = nullptr;
    for (const auto &arg : func->inputArguments)
      if (!isDeclaredOutput(arg) &&
          (!firstInput || position(arg) < position(*firstInput)))
        firstInput = &arg;
    if (!firstInput)
      return success();
    // Blame the first output that comes after an input.
    const KernelDeclOperandAdaptor *lateOutput = nullptr;
    auto consider = [&](const KernelDeclOperandAdaptor &arg) {
      if (position(arg) > position(*firstInput) &&
          (!lateOutput || position(arg) < position(*lateOutput)))
        lateOutput = &arg;
    };
    for (const auto &arg : func->outputArguments)
      consider(arg);
    for (const auto &arg : func->inputArguments)
      if (isDeclaredOutput(arg))
        consider(arg);
    if (lateOutput) {
      emitMojoArgumentError(decl, kind, *lateOutput)
          << "Outputs must appear before inputs in the function signature: "
          << "output '" << lateOutput->sourceDescriptor->sourceName.getValue()
          << "' comes after input '"
          << firstInput->sourceDescriptor->sourceName.getValue() << "'";
      return failure();
    }
    return success();
  };

  if (outputsBeforeInputs(adaptor.executeFunction, KernelFunctionKind::Execute)
          .failed() ||
      outputsBeforeInputs(adaptor.buildGraphFunction,
                          KernelFunctionKind::BuildGraph)
          .failed())
    return failure();
  return success();
}

/// Each kernel function variant has at most one launch arg
/// (DeviceContext / DeviceContextArray on execute / shape, DeviceGraphBuilder
/// on build_graph) and the launch arg must come last in the signature.
LogicalResult declVerifyContextsAppearLastInSignature(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  const auto launchArgsAreLast =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func,
              KernelFunctionKind kind, llvm::StringRef argLower,
              llvm::StringRef argTitle) -> LogicalResult {
    if (!func.has_value())
      return success();
    const KernelDeclOperandAdaptor *launchArg = nullptr;
    int64_t indexOfLastArgument = std::numeric_limits<int64_t>::min();
    for (const auto &arg : func->inputArguments) {
      if (arg.isLaunchArgument()) {
        if (launchArg) {
          emitMojoArgumentError(decl, kind, arg)
              << "There can be at most one " << argLower;
          return failure();
        }
        launchArg = &arg;
      }
      indexOfLastArgument = std::max<int64_t>(indexOfLastArgument,
                                              arg.sourceDescriptor->position);
    }
    if (launchArg &&
        static_cast<int64_t>(launchArg->sourceDescriptor->position) <
            indexOfLastArgument) {
      emitMojoArgumentError(decl, kind, *launchArg)
          << argTitle
          << " argument must be the last argument "
             "in the function signature";
      return failure();
    }
    return success();
  };

  if (launchArgsAreLast(adaptor.executeFunction, KernelFunctionKind::Execute,
                        "context", "Context")
          .failed() ||
      launchArgsAreLast(shapeFunction, KernelFunctionKind::Shape, "context",
                        "Context")
          .failed() ||
      launchArgsAreLast(adaptor.buildGraphFunction,
                        KernelFunctionKind::BuildGraph, "graph builder",
                        "Graph builder")
          .failed())
    return failure();
  return success();
}

LogicalResult declVerifyOpaqueArgumentsArePassedByReference(
    KernelDeclOp decl, KernelDeclAdaptor &adaptor,
    std::optional<KernelDeclFunctionAdaptor> shapeFunction) {
  auto opaquesByReference =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value())
      return success();
    auto allArgs = llvm::concat<const KernelDeclOperandAdaptor>(
        func->outputArguments, func->inputArguments);
    for (const auto &arg : allArgs) {
      // Copy sig to call the non-const getArgConvention.
      auto sig = func->attr.getSignature();
      auto convention = sig.getArgConvention(arg.sourceDescriptor->position);
      if (arg.isOpaqueType() && (convention == KGEN::ArgConvention::OwnedMem ||
                                 convention == KGEN::ArgConvention::DeinitMem ||
                                 convention == KGEN::ArgConvention::OwnedReg)) {
        decl->emitError()
            << "Argument convention for argument at position "
            << arg.sourceDescriptor->position
            << " is invalid: Owned arguments not supported for custom types "
               "in kernels";
        return failure();
      }
      // Tensors must be passed by const reference (`read`/`read_mem`):
      // their mutability is in the IOSpec type, not the calling convention.
      // Trait-bound tensors are left to `declVerifyTensorArgsAreBorrowed` and
      // its trait-specific diagnostic.
      if (arg.isTensorType() && !arg.isTraitTensorType() &&
          convention != KGEN::ArgConvention::ImmMem &&
          convention != KGEN::ArgConvention::ImmReg) {
        decl->emitError()
            << "Only the borrowed argument (read) convention is supported "
               "for tensor arguments (argument at position "
            << arg.sourceDescriptor->position << " uses '"
            << KGEN::stringifyArgConvention(convention) << "')";
        return failure();
      }
    }
    return success();
  };

  if (opaquesByReference(adaptor.executeFunction).failed() ||
      opaquesByReference(shapeFunction).failed() ||
      opaquesByReference(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

// Trait-bound tensor arguments carry their role and mutability in the distilled
// contract, not the calling convention, so they must be borrowed
// (`read`/`read_mem`).
static bool isTraitBoundTensorArg(Type argType) {
  return !getTensorArgTraitSymbols(argType).empty();
}

LogicalResult declVerifyTensorArgsAreBorrowed(
    KernelDeclOp decl, KernelDeclAdaptor &adaptor,
    std::optional<KernelDeclFunctionAdaptor> shapeFunction) {
  auto borrowed =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value())
      return success();
    // Copy sig to call the non-const getArgConvention.
    auto sig = func->attr.getSignature();
    for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
      if (!isTraitBoundTensorArg(sig.getArgument(i)))
        continue;
      auto convention = sig.getArgConvention(i);
      if (convention != KGEN::ArgConvention::ImmMem &&
          convention != KGEN::ArgConvention::ImmReg) {
        decl->emitError()
            << "Only the borrowed argument (read) convention is supported for "
               "trait-bound tensor arguments (argument at position "
            << i << " uses '" << KGEN::stringifyArgConvention(convention)
            << "')";
        return failure();
      }
    }
    return success();
  };

  if (borrowed(adaptor.executeFunction).failed() ||
      borrowed(shapeFunction).failed() ||
      borrowed(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

// Per-function-kind policy for trait-bound tensor arguments: `execute`,
// `build_graph`, and the declarative `execute_device` /
// `execute_configuration` accept them. Kind-specific rules go here (e.g.
// build_graph forbidding `TileTensorable`); shape functions, primitives and
// universal well-formedness are checked at import.
LogicalResult declVerifyTraitArgFunctionKinds(KernelDeclOp decl,
                                              KernelDeclAdaptor &adaptor) {
  auto rejectTraitArgs = [&decl](std::optional<KernelDeclFunctionAdaptor> &func,
                                 KernelFunctionKind kind) -> LogicalResult {
    if (!func.has_value())
      return success();
    auto sig = func->attr.getSignature();
    bool ok = true;
    for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
      if (isTraitBoundTensorArg(sig.getArgument(i))) {
        decl->emitError()
            << "trait-bound tensor arguments are not supported on "
            << stringifyKernelFunctionKind(kind) << " functions yet (only `"
            << stringifyKernelFunctionKind(KernelFunctionKind::Execute)
            << "`, `"
            << stringifyKernelFunctionKind(KernelFunctionKind::BuildGraph)
            << "`, `"
            << stringifyKernelFunctionKind(KernelFunctionKind::ExecuteDevice)
            << "` and `"
            << stringifyKernelFunctionKind(
                   KernelFunctionKind::ExecuteConfiguration)
            << "`)";
        ok = false;
      }
    }
    return ok ? success() : failure();
  };

  bool ok = true;
  if (rejectTraitArgs(adaptor.elementwiseFunction,
                      KernelFunctionKind::Elementwise)
          .failed())
    ok = false;
  if (rejectTraitArgs(adaptor.tileElementwiseFunction,
                      KernelFunctionKind::TileElementwise)
          .failed())
    ok = false;
  return ok ? success() : failure();
}

LogicalResult declVerifyAtMostOneVariadicInputAndOutput(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  // TODO: (GEX-1994) Remove this.
  if (adaptor.registeredOpName.strref().starts_with("mo."))
    return success();

  const auto atMostOneVariadic =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();
    if (llvm::count_if(func->inputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      decl->emitError()
          << "Kernel function can have at most one variadic input tensors";
      return failure();
    }
    if (llvm::count_if(func->outputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      decl->emitError()
          << "Kernel function can have at most one variadic output tensors";
      return failure();
    }
    return success();
  };

  if (atMostOneVariadic(adaptor.executeFunction).failed() ||
      atMostOneVariadic(shapeFunction).failed() ||
      atMostOneVariadic(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyExecuteNotReturningOpaqueAndThrows(KernelDeclOp decl,
                                             KernelDeclAdaptor &adaptor) {
  auto notOpaqueAndThrows =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value() || !func->outputResult.has_value())
      return success();
    auto sig = func->attr.getSignature();
    if (func->outputResult->isOpaqueType() && sig.isThrows()) {
      decl->emitError() << "Functions that raise and return a custom type are "
                           "not yet supported";
      return failure();
    }
    return success();
  };

  if (notOpaqueAndThrows(adaptor.executeFunction).failed() ||
      notOpaqueAndThrows(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult declVerifyHasEntryPoint(KernelDeclOp decl,
                                      const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value() &&
      !adaptor.elementwiseFunction.has_value() && !adaptor.isReduction()) {
    decl->emitError() << "The kernel must have an entry point named execute";
    return failure();
  }
  return success();
}

// A kernel reads its device through its `DeviceContext` argument's `target`, so
// the graph compiler no longer infers a `target: StaticString` parameter.
LogicalResult declVerifyNoTargetParameter(KernelDeclOp decl,
                                          const KernelDeclAdaptor &adaptor) {
  bool failed = false;
  auto check =
      [&](mlir::ArrayAttr params, mlir::Location loc, llvm::StringRef where) {
        if (!params)
          return;
        for (mlir::Attribute attr : params) {
          auto param = mlir::cast<ParamDeclAttr>(attr);
          if (demangleParameterName(param.getName().strref()) !=
              MOGG::kMoggTargetParameterName)
            continue;
          mlir::emitError(loc)
              << where << " declares a `target` parameter, which is no longer "
              << "supported; take a `ctx: DeviceContext` argument and use "
                 "`ctx.target` instead";
          failed = true;
        }
      };

  check(adaptor.structParams, decl.getMojoStructLocation(),
        "the kernel struct");
  auto checkFunction =
      [&](const std::optional<KernelDeclFunctionAdaptor> &function,
          KernelFunctionKind kind) {
        if (function && function->attr)
          check(function->attr.getParams(), decl.getMojoFunctionLocation(kind),
                stringifyKernelFunctionKind(kind));
      };
  checkFunction(adaptor.executeFunction, KernelFunctionKind::Execute);
  checkFunction(adaptor.elementwiseFunction, KernelFunctionKind::Elementwise);
  checkFunction(adaptor.tileElementwiseFunction,
                KernelFunctionKind::TileElementwise);
  checkFunction(adaptor.buildGraphFunction, KernelFunctionKind::BuildGraph);
  checkFunction(adaptor.executeDeviceFunction,
                KernelFunctionKind::ExecuteDevice);
  checkFunction(adaptor.executeConfigurationFunction,
                KernelFunctionKind::ExecuteConfiguration);
  return failed ? failure() : success();
}

// Whether the generated launch can pass `operand` to a declarative entry point:
// a trait-bound tensor/buffer crosses as its `TileTensor` projection and is
// rebuilt on the device, a fused tensor crosses as itself, and a scalar
// crosses as-is.
static bool isDeclarativeOperand(const KernelDeclOperandAdaptor &operand) {
  if (std::holds_alternative<KernelDeclScalarAdaptor>(operand.underlyingType))
    return true;
  const auto *tensor =
      std::get_if<KernelDeclTensorAdaptor>(&operand.underlyingType);
  return tensor && tensor->traitContract;
}

// `execute_device` and `execute_configuration` come as a pair over the same
// operands: the generated launch passes each the call's operands,
// `execute_configuration` on the host and `execute_device` on the device.
LogicalResult declVerifyDeclarative(KernelDeclOp decl,
                                    const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeDeviceFunction && !adaptor.executeConfigurationFunction)
    return success();
  // A package's kernels are verified together, so each error names its kernel.
  auto error = [&] {
    return decl->emitError() << "kernel '" << decl.getRegistration() << "': ";
  };
  StringRef executeName =
      stringifyKernelFunctionKind(KernelFunctionKind::Execute);
  StringRef deviceName =
      stringifyKernelFunctionKind(KernelFunctionKind::ExecuteDevice);
  StringRef configurationName =
      stringifyKernelFunctionKind(KernelFunctionKind::ExecuteConfiguration);
  if (adaptor.executeDeviceFunction && decl.getExecute()) {
    return error() << "a kernel has either '" << executeName << "' or '"
                   << deviceName << "'";
  }
  if (!adaptor.executeDeviceFunction || !adaptor.executeConfigurationFunction) {
    return error() << "a declarative kernel needs both '" << deviceName
                   << "' and '" << configurationName << "'";
  }

  const KernelDeclFunctionAdaptor &executeDevice =
      *adaptor.executeDeviceFunction;
  if (executeDevice.attr.getSignature().isThrows())
    return error() << "'" << deviceName
                   << "' runs on the device and must not raise";
  if (executeDevice.outputResult)
    return error() << "'" << deviceName << "' must not return a value";

  // `execute_configuration` takes its operands by `Tensor`, which has no io
  // role, so all of them count as inputs; compare the operands in order.
  const KernelDeclFunctionAdaptor &executeConfiguration =
      *adaptor.executeConfigurationFunction;
  auto operandsOf = [](const KernelDeclFunctionAdaptor &fn) {
    SmallVector<const KernelDeclOperandAdaptor *> operands;
    for (const KernelDeclOperandAdaptor &arg : fn.outputArguments)
      operands.push_back(&arg);
    for (const KernelDeclOperandAdaptor &arg : fn.inputArguments)
      operands.push_back(&arg);
    return operands;
  };
  auto deviceOperands = operandsOf(executeDevice);
  auto configurationOperands = operandsOf(executeConfiguration);
  auto checkOperands = [&](StringRef function,
                           ArrayRef<const KernelDeclOperandAdaptor *> operands)
      -> LogicalResult {
    for (auto [i, operand] : llvm::enumerate(operands)) {
      if (!isDeclarativeOperand(*operand)) {
        return error() << "'" << function << "' operand " << i
                       << " must be a trait-bound tensor or a scalar";
      }
    }
    return success();
  };
  if (failed(checkOperands(deviceName, deviceOperands)) ||
      failed(checkOperands(configurationName, configurationOperands)))
    return failure();

  if (deviceOperands.size() != configurationOperands.size()) {
    return error() << "'" << configurationName << "' takes "
                   << configurationOperands.size() << " argument(s) but '"
                   << deviceName << "' takes " << deviceOperands.size()
                   << "; both take the kernel's operands";
  }
  for (size_t i = 0, e = deviceOperands.size(); i < e; ++i) {
    bool deviceTensor = deviceOperands[i]->isTensorType();
    if (deviceTensor != configurationOperands[i]->isTensorType()) {
      return error() << "operand " << i << " is a "
                     << (deviceTensor ? "tensor" : "scalar") << " in '"
                     << deviceName << "' but a "
                     << (deviceTensor ? "scalar" : "tensor") << " in '"
                     << configurationName << "'";
    }
  }
  return success();
}

// The rules `algorithm.reduce_op.ReduceOp` documents for monoids the GC
// instantiates.
LogicalResult declVerifyReduction(KernelDeclOp decl,
                                  const KernelDeclAdaptor &adaptor) {
  KernelReductionAttr reduction = *adaptor.reduction;
  llvm::StringRef name = adaptor.registeredOpName.strref();
  if (adaptor.executeFunction || adaptor.elementwiseFunction ||
      adaptor.tileElementwiseFunction || adaptor.buildGraphFunction)
    return decl->emitError()
           << "reduction '" << name << "' must not define a kernel entry point";
  llvm::StringRef width = reduction.getWidthParam().strref();
  if (width.empty())
    return decl->emitError()
           << "reduction '" << name
           << "': the `width` alias must name one of the struct's parameters";
  if (reduction.getOutputs().empty())
    return decl->emitError()
           << "reduction '" << name
           << "' has no output fields (SIMD fields whose width is `" << width
           << "`)";

  // Every parameter must be the width, an output's dtype, or defaulted.
  llvm::StringSet<> inferable;
  inferable.insert(width);
  for (mlir::Attribute attr : reduction.getOutputs()) {
    mlir::Attribute dtype =
        mlir::cast<KernelReductionOutputAttr>(attr).getDtype();
    if (auto ref = mlir::dyn_cast<ParamDeclRefAttr>(SugarAttr::strip(dtype)))
      inferable.insert(demangleParameterName(ref.getName().strref()));
  }
  PogListAttr pogs = reduction.getSignature().getParamListAttrs();
  for (auto [i, attr] : llvm::enumerate(adaptor.structParams)) {
    llvm::StringRef param = demangleParameterName(
        mlir::cast<ParamDeclAttr>(attr).getName().strref());
    bool hasDefault = i < pogs.getPogs().size() && pogs.getDefault(i);
    if (!inferable.contains(param) && !hasDefault)
      return decl->emitError()
             << "reduction '" << name << "': parameter '" << param
             << "' is neither the `width` parameter, an output's dtype, nor "
                "defaulted";
  }
  return success();
}

LogicalResult declVerifyElementwiseFunction(KernelDeclOp decl,
                                            const KernelDeclAdaptor &adaptor) {
  if (!adaptor.elementwiseFunction.has_value())
    return success();

  const KernelDeclFunctionAdaptor &elemwiseFn = *adaptor.elementwiseFunction;

  // Copy sig to call non-const isThrows.
  auto sig = elemwiseFn.attr.getSignature();
  if (sig.isThrows())
    return decl->emitError() << "Elementwise kernels can't throw";

  if (!elemwiseFn.outputArguments.empty())
    return decl->emitError() << "Elementwise kernels must return an SIMD value";
  if (!elemwiseFn.outputResult.has_value() ||
      !elemwiseFn.outputResult->isScalarType())
    return decl->emitError() << "Elementwise kernels must return an SIMD value";

  size_t operandsCount = 0;
  for (size_t i = 0, inputsCount = elemwiseFn.inputArguments.size();
       i < inputsCount; ++i) {
    // The last argument might be an IndexList. Skip it — it is an
    // elementwise-only concept.
    if (i + 1 == inputsCount && elemwiseFn.takesIndexList())
      continue;
    ++operandsCount;
    if (!elemwiseFn.inputArguments[i].isScalarType())
      return decl->emitError() << "Elementwise kernels must have SIMD operands";
  }

  if (!operandsCount)
    return decl->emitError()
           << "Elementwise kernels must have at least one operand";

  return success();
}

LogicalResult
declVerifyTileElementwiseFunction(KernelDeclOp decl,
                                  const KernelDeclAdaptor &adaptor) {
  if (!adaptor.tileElementwiseFunction.has_value())
    return success();

  const KernelDeclFunctionAdaptor &fn = *adaptor.tileElementwiseFunction;

  auto sig = fn.attr.getSignature();
  if (sig.isThrows())
    return decl->emitError() << "Tile elementwise kernels can't throw";

  if (!fn.outputArguments.empty())
    return decl->emitError()
           << "Tile elementwise kernels must return a TileTensor value";
  if (!fn.outputResult.has_value() || !fn.outputResult->isTileType())
    return decl->emitError()
           << "Tile elementwise kernels must return a TileTensor value";

  size_t operandsCount = 0;
  for (const auto &arg : fn.inputArguments) {
    ++operandsCount;
    if (!arg.isTileType())
      return decl->emitError()
             << "Tile elementwise kernels must have TileTensor operands";
  }

  if (!operandsCount)
    return decl->emitError()
           << "Tile elementwise kernels must have at least one operand";

  return success();
}

//===----------------------------------------------------------------------===//
// Cross-kernel verifiers (consistency across multiple registrations of the
// same op name)
//===----------------------------------------------------------------------===//

enum class KernelTypeTag { Execute, Elementwise, Reduction };

static KernelTypeTag getKernelType(const KernelDeclAdaptor &adaptor) {
  if (adaptor.isReduction())
    return KernelTypeTag::Reduction;
  if (adaptor.isElementwise() || adaptor.isTileElementwise())
    return KernelTypeTag::Elementwise;
  if (adaptor.executeFunction.has_value())
    return KernelTypeTag::Execute;
  llvm_unreachable("kernel has no recognised entry point; "
                   "declVerifyHasEntryPoint should have caught this");
}

static llvm::StringRef kernelTypeName(KernelTypeTag t) {
  switch (t) {
  case KernelTypeTag::Execute:
    return "execute";
  case KernelTypeTag::Elementwise:
    return "elementwise";
  case KernelTypeTag::Reduction:
    return "reduction";
  }
  llvm_unreachable("unknown KernelTypeTag");
}

static llvm::SmallVector<KernelDeclOperandAdaptor>
getNonLaunchInputs(const KernelDeclFunctionAdaptor &fn) {
  llvm::SmallVector<KernelDeclOperandAdaptor> result;
  for (const KernelDeclOperandAdaptor &arg : fn.inputArguments)
    if (!arg.isLaunchArgument())
      result.push_back(arg);
  return result;
}

// Return the semantic result of a kernel function, or nullopt if the function
// uses `raises` (in which case outputResult is a raises ABI artifact, not a
// real return value).
static std::optional<KernelDeclOperandAdaptor>
getSemanticResult(const KernelDeclFunctionAdaptor &fn) {
  if (fn.attr.getSignature().isThrows())
    return std::nullopt;
  return fn.outputResult;
}

/// Verify cross-kernel consistency for every registration identity — an op
/// name and a set of devices — with multiple declarations. Two registrations
/// have the same identity if their platforms apply to exactly the
/// same devices, however the platforms are written. ImportPackages rejects
/// such a pair within one provenance group, so a group here is a user
/// registration that replaces a built-in on the same devices. For each group,
/// decls[0] is the reference (ImportPackages inserts built-ins first, so it is
/// the built-in when one exists).
///
/// Decls differing in device scope are grouped apart: `getResolvedKernel`
/// picks among them by device, so they need not agree (e.g. an accelerator
/// `execute` kernel for an elementwise built-in).
///
/// Per registration, checks in order:
///   1. Kernel type must match (execute / elementwise / view).
///   2. If execute: output count, non-launch input count, and per-operand
///      types must all match (fusion variants are treated as equivalent).
LogicalResult crossVerifyKernels(ModuleOp moduleOp) {
  bool failed = false;
  llvm::MapVector<mlir::StringAttr, llvm::SmallVector<KernelDeclOp>> byName;
  for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>())
    byName[decl.getRegistrationAttr()].push_back(decl);

  llvm::SmallVector<llvm::SmallVector<KernelDeclOp>> groups;
  for (auto &[name, decls] : byName) {
    if (decls.size() < 2)
      continue;
    llvm::SmallVector<
        std::pair<PlatformPredicate, llvm::SmallVector<KernelDeclOp>>>
        classes;
    for (KernelDeclOp decl : decls) {
      ErrorOr<PlatformPredicate> platform =
          getRegistrationPlatformPredicate(decl);
      if (platform.isError()) {
        decl->emitError() << "invalid platform: " << platform.getError();
        failed = true;
        continue;
      }
      auto *it = classes.begin();
      for (; it != classes.end(); ++it)
        if (comparePlatformPredicates(it->first, *platform).order ==
            PlatformPredicateOrder::Equivalent)
          break;
      if (it == classes.end())
        classes.push_back({platform.takeValue(), {decl}});
      else
        it->second.push_back(decl);
    }
    for (auto &[platform, members] : classes)
      if (members.size() >= 2)
        groups.push_back(std::move(members));
  }

  for (llvm::SmallVector<KernelDeclOp> &decls : groups) {
    mlir::StringAttr regName = decls.front().getRegistrationAttr();

    KernelDeclAdaptor refAdaptor(decls[0]);
    KernelTypeTag refTag = getKernelType(refAdaptor);
    bool refIsBuiltin = !decls[0].getIsUserDefined();

    // Pre-compute reference signature (used when refTag == Execute or
    // Elementwise).
    const llvm::SmallVector<KernelDeclOperandAdaptor> *refOutputs = nullptr;
    llvm::SmallVector<KernelDeclOperandAdaptor> refInputs;
    std::optional<KernelDeclOperandAdaptor> refOutputResult;
    if (refTag == KernelTypeTag::Execute) {
      refOutputs = &refAdaptor.executeFunction->outputArguments;
      refInputs = getNonLaunchInputs(*refAdaptor.executeFunction);
      refOutputResult = getSemanticResult(*refAdaptor.executeFunction);
    } else if (refTag == KernelTypeTag::Elementwise) {
      refOutputs = &refAdaptor.elementwiseFunction->outputArguments;
      refInputs = getNonLaunchInputs(*refAdaptor.elementwiseFunction);
      refOutputResult = getSemanticResult(*refAdaptor.elementwiseFunction);
    }

    for (size_t i = 1; i < decls.size(); ++i) {
      KernelDeclAdaptor adaptor(decls[i]);

      // 1. Kernel type check.
      KernelTypeTag tag = getKernelType(adaptor);
      if (tag != refTag) {
        if (refIsBuiltin) {
          decls[i]->emitError()
              << "Kernel \"" << regName.getValue()
              << "\" type mismatch: built-in uses " << kernelTypeName(refTag)
              << ", but this registration uses " << kernelTypeName(tag) << "; "
              << kernelTypeName(refTag) << " method expected";
        } else {
          decls[i]->emitError()
              << "Kernel \"" << regName.getValue()
              << "\" type mismatch: this registration uses "
              << kernelTypeName(tag) << " but a previous registration uses "
              << kernelTypeName(refTag) << "; " << kernelTypeName(refTag)
              << " method expected";
        }
        failed = true;
        continue;
      }

      // 2. Signature check (execute and elementwise; skip view / build-graph).
      if (refTag != KernelTypeTag::Execute &&
          refTag != KernelTypeTag::Elementwise)
        continue;

      llvm::StringRef sigKind;
      const llvm::SmallVector<KernelDeclOperandAdaptor> *outputs = nullptr;
      llvm::SmallVector<KernelDeclOperandAdaptor> inputs;
      std::optional<KernelDeclOperandAdaptor> outputResult;
      if (refTag == KernelTypeTag::Execute) {
        sigKind = "Execute";
        outputs = &adaptor.executeFunction->outputArguments;
        inputs = getNonLaunchInputs(*adaptor.executeFunction);
        outputResult = getSemanticResult(*adaptor.executeFunction);
      } else {
        sigKind = "Elementwise";
        outputs = &adaptor.elementwiseFunction->outputArguments;
        inputs = getNonLaunchInputs(*adaptor.elementwiseFunction);
        outputResult = getSemanticResult(*adaptor.elementwiseFunction);
      }

      // Count mismatch.
      if (outputs->size() != refOutputs->size() ||
          inputs.size() != refInputs.size() ||
          outputResult.has_value() != refOutputResult.has_value()) {
        if (refIsBuiltin) {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": " << regName.getValue() << " expects "
              << refOutputs->size() << " output(s), " << refInputs.size()
              << " input(s), and " << (refOutputResult.has_value() ? 1 : 0)
              << " result(s), but this registration has " << outputs->size()
              << " output(s), " << inputs.size() << " input(s), and "
              << (outputResult.has_value() ? 1 : 0) << " result(s)";
        } else {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": previous registration has " << refOutputs->size()
              << " output(s), " << refInputs.size() << " input(s), and "
              << (refOutputResult.has_value() ? 1 : 0)
              << " result(s), but this registration has " << outputs->size()
              << " output(s), " << inputs.size() << " input(s), and "
              << (outputResult.has_value() ? 1 : 0) << " result(s)";
        }
        failed = true;
        continue;
      }

      // `execute_device` takes trait-bound operands, which never compare equal
      // to an `execute`'s tensor types, so a declarative kernel beside an
      // `execute` kernel is checked by count only.
      if (adaptor.isDeclarative() != refAdaptor.isDeclarative())
        continue;

      // Per-operand type check.
      auto emitOperandError = [&](size_t pos, llvm::StringRef kind,
                                  const KernelDeclOperandAdaptor &refOp,
                                  const KernelDeclOperandAdaptor &op) {
        std::string refStr, str;
        llvm::raw_string_ostream(refStr) << refOp;
        llvm::raw_string_ostream(str) << op;
        if (refIsBuiltin) {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": " << kind << " " << pos
              << " type differs: " << regName.getValue() << " expects "
              << refStr << ", but this registration has " << str;
        } else {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": " << kind << " " << pos
              << " type differs: previous registration has " << refStr
              << ", but this registration has " << str;
        }
        failed = true;
      };

      for (size_t j = 0; j < refOutputs->size(); ++j)
        if ((*refOutputs)[j] != (*outputs)[j])
          emitOperandError(j, "output", (*refOutputs)[j], (*outputs)[j]);
      for (size_t j = 0; j < refInputs.size(); ++j)
        if (refInputs[j] != inputs[j])
          emitOperandError(j, "input", refInputs[j], inputs[j]);
      if (refOutputResult.has_value() && outputResult.has_value() &&
          *refOutputResult != *outputResult)
        emitOperandError(0, "result", *refOutputResult, *outputResult);
    }
  }
  return failed ? failure() : success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct VerifyKernelsPass : public impl::VerifyKernelsBase<VerifyKernelsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MojoRegistry registry(moduleOp);

    bool failed = false;
    for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>()) {
      std::optional<KernelDeclFunctionAdaptor> shapeFnAdaptor;
      if (registry.containsShapeFunction(decl.getRegistration()))
        shapeFnAdaptor = KernelDeclFunctionAdaptor(
            registry.lookupShapeFunction(decl.getRegistration()));
      KernelDeclAdaptor adaptor(decl);

      // The declarative check runs first, so that a lone
      // `execute_configuration` reports its missing `execute_device` rather
      // than a missing `execute`.
      if (setFlagIfFailed(failed, declVerifyDeclarative, decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyHasEntryPoint, decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyNoTargetParameter, decl, adaptor)
              .failed())
        continue;

      // A reduction's entry points are the monoid's own methods.
      if (adaptor.isReduction()) {
        (void)setFlagIfFailed(failed, declVerifyReduction, decl, adaptor);
        continue;
      }

      // Elementwise-only kernels: verify elementwise function(s).
      if (!adaptor.executeFunction.has_value()) {
        if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
        if (setFlagIfFailed(failed, declVerifyTileElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
        continue;
      }

      if (setFlagIfFailed(failed, declVerifyBuildGraphTakesTraitArguments, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyShapeFunctionReturnsShape, decl,
                          shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed,
                          declVerifyShapeFunctionInputsMatchExecuteFunction,
                          decl, adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyBuildGraphHasBuilder, decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteFunctionDoesNotReturnATensor,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyOutputsAppearBeforeInputs, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyContextsAppearLastInSignature, decl,
                          adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteMatchesBuildGraph, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyStableTensorArguments, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyOpaqueArgumentsArePassedByReference,
                          decl, adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyTensorArgsAreBorrowed, decl,
                          adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyTraitArgFunctionKinds, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyAtMostOneVariadicInputAndOutput,
                          decl, adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteNotReturningOpaqueAndThrows,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl, adaptor)
              .failed())
        continue;
    }

    // Cross-kernel consistency checks.
    (void)setFlagIfFailed(failed, crossVerifyKernels, moduleOp);

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
