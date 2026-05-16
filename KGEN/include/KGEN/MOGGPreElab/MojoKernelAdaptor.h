//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_KERNEL_ADAPTOR_H
#define MOJO_KERNEL_ADAPTOR_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"

#include <type_traits>
#include <variant>

template <typename T, typename... Ts>
constexpr bool is_one_of = (std::is_same_v<T, Ts> || ...);

template <typename... TargetTypes, typename Variant, typename Func>
void partial_visit(Func &&func, Variant &var) {
  std::visit(
      [&](auto &val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (is_one_of<T, TargetTypes...>)
          func(val);
      },
      var);
}

namespace M::KGEN::MOGGPreElab {

struct TensorOperandAdaptor {
  static constexpr StringLiteral typeName = MOJO_DPS_TENSOR_TYPE_NAME;
  // If the tensor is a mutable input or not.
  bool mut;
  bool fused;
  IOSpec ioSpec;
  TensorOperandAdaptor() = default;
  TensorOperandAdaptor(IOSpec ioSpec)
      : mut(isMutableIOSpec(ioSpec)), fused(isFusableIOSpec(ioSpec)),
        ioSpec(ioSpec) {}

  bool operator==(const TensorOperandAdaptor &other) const {
    // Not comparing fused because it can't be specified anywhere except in Mojo
    // (nothing to compare it to).
    return mut == other.mut;
  }
};

struct VariadicTensorOperandAdaptor : TensorOperandAdaptor {
  static constexpr StringLiteral typeName = MOJO_VARIADIC_TENSORS_NAME;
  bool operator==(const VariadicTensorOperandAdaptor &other) const {
    return mut == other.mut;
  }
};

struct ScalarOperandAdaptor {
  static constexpr StringLiteral typeName = MOJO_INTERNAL_DPS_SIMD_TYPE_NAME;
  bool operator==(const ScalarOperandAdaptor &other) const { return true; }
};

struct DevicesContextPtrOperandAdaptor {
  static constexpr StringLiteral typeName =
      MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_PTR_TYPE_NAME;
  bool operator==(const DevicesContextPtrOperandAdaptor &other) const {
    return true;
  }
};

struct DevicesContextPtrListOperandAdaptor {
  static constexpr StringLiteral typeName =
      MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_PTR_LIST_TYPE_NAME;
  bool operator==(const DevicesContextPtrListOperandAdaptor &other) const {
    return true;
  }
};

struct OpaqueOperandAdaptor {
  StringRef typeName;
  bool operator==(const OpaqueOperandAdaptor &other) const {
    if (typeName == other.typeName)
      return true;

    /// FIXME(GEX-2036): This is a gross hack. In `MOGGPreElab`, we maintain
    /// module names. In other parts of the stack, we use the leaf name.
    /// Therefore, we check if one name ends with the other. A proper fix would
    /// be to use the same name at all times.
    if (typeName.ends_with(other.typeName) ||
        other.typeName.ends_with(typeName))
      return true;

    return false;
  }
};

struct UnsupportedOperandAdaptor {
  bool operator==(const UnsupportedOperandAdaptor &other) const {
    return false;
  }
};

struct MojoKernelOperandSourceDescriptor {
  // The name of the variable in the original Mojo code.
  StringRef sourceName;
  // Its position in the function.
  uint64_t position;
  // If it was a return value that was promoted to a by ref result by the Mojo
  // compiler.
  bool isByRefResult;
};

using MojoKernelOperandVariant = std::variant<
    TensorOperandAdaptor, VariadicTensorOperandAdaptor, ScalarOperandAdaptor,
    OpaqueOperandAdaptor, DevicesContextPtrOperandAdaptor,
    DevicesContextPtrListOperandAdaptor, UnsupportedOperandAdaptor>;

struct MojoKernelOperandAdaptor {
  MojoKernelOperandAdaptor() = default;

  // The source information which is optional in the case of a return value.
  std::optional<MojoKernelOperandSourceDescriptor> sourceDescriptor;
  // A union between all kinds of supported operands.
  MojoKernelOperandVariant underlyingType;

  MojoKernelOperandAdaptor(std::optional<uint64_t> positionInFunction,
                           StringAttr typeName, ArrayAttr argumentSourceNames,
                           ArrayAttr argsIoSpecs, bool isByRefResult = false,
                           bool promoteSIMDToFusedTensor = false);

  static MojoKernelOperandAdaptor buildElementwiseOutputOperand();

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    if (sourceDescriptor.has_value())
      os << nesting << *sourceDescriptor;

    os << underlyingType;

    return os;
  }

  bool operator==(const MojoKernelOperandAdaptor &other) const {
    if (underlyingType.index() != other.underlyingType.index())
      return false;

    return std::visit(
        [&](auto &&arg) { return underlyingType == other.underlyingType; },
        underlyingType);
  }

  bool operator!=(const MojoKernelOperandAdaptor &other) const {
    return !(*this == other);
  }

  bool isDpsOutput() const {
    return sourceDescriptor.has_value() && !sourceDescriptor->isByRefResult;
  }

  bool isVariadicTensorType() const {
    return std::holds_alternative<VariadicTensorOperandAdaptor>(underlyingType);
  }

  bool isOpaqueType() const {
    return std::holds_alternative<OpaqueOperandAdaptor>(underlyingType);
  }

  bool isUnsupportedType() const {
    return std::holds_alternative<UnsupportedOperandAdaptor>(underlyingType);
  }

  bool isContextType() const {
    return std::holds_alternative<DevicesContextPtrOperandAdaptor>(
               underlyingType) ||
           std::holds_alternative<DevicesContextPtrListOperandAdaptor>(
               underlyingType);
  }

  bool isVariadicContextType() const {
    return std::holds_alternative<DevicesContextPtrListOperandAdaptor>(
        underlyingType);
  }

  bool isScalarType() const {
    return std::holds_alternative<ScalarOperandAdaptor>(underlyingType);
  }

  bool isTensorType() const {
    return std::holds_alternative<TensorOperandAdaptor>(underlyingType) ||
           std::holds_alternative<VariadicTensorOperandAdaptor>(underlyingType);
  }

  bool isFusedTensorType() const {
    if (!isTensorType())
      return false;

    bool fused;
    partial_visit<TensorOperandAdaptor, VariadicTensorOperandAdaptor>(
        [&](auto &&obj) { fused = obj.fused; }, underlyingType);

    return fused;
  }

  bool isTensorWithIOSpec(IOSpec spec) const {
    if (!isTensorType())
      return false;

    bool match = false;
    partial_visit<TensorOperandAdaptor, VariadicTensorOperandAdaptor>(
        [&](auto &&obj) { match = (spec == obj.ioSpec); }, underlyingType);

    return match;
  }

  bool isMutableTensorType() const {
    if (!isTensorType())
      return false;

    bool isMutable;
    partial_visit<TensorOperandAdaptor, VariadicTensorOperandAdaptor>(
        [&](auto &&obj) { isMutable = obj.mut; }, underlyingType);

    return isMutable;
  }

  bool isOpaqueWithTypeName(StringRef name) const {
    if (!std::holds_alternative<OpaqueOperandAdaptor>(underlyingType))
      return false;

    return std::get<OpaqueOperandAdaptor>(underlyingType).typeName == name;
  }
};

struct MojoKernelFunctionAdaptor {
  // The underlying LIT or KGEN function.
  // Marked mutable because getSymName is not const for some reason.
  mutable KGEN::LIT::FnOp mojoCode;
  // Input arguments.
  SmallVector<MojoKernelOperandAdaptor> inputArguments;
  // Output arguments (DPS).
  SmallVector<MojoKernelOperandAdaptor> outputArguments;
  // Output result which could be an argument if the function throws.
  std::optional<MojoKernelOperandAdaptor> outputResult;

  MojoKernelFunctionAdaptor(KGEN::LIT::FnOp op) : mojoCode(op) {
    auto argumentTypesNames = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::MOGG_ARG_TYPE_NAMES);
    auto resultTypeNameAttr =
        mojoCode->getAttr(KGEN::MOGGPreElab::MOGG_RESULT_TYPE_NAME);
    auto numberOfOutputArgumentsAttr =
        mojoCode->template getAttrOfType<IntegerAttr>(
            KGEN::MOGGPreElab::kMOGGNumDPSOutputs);
    auto argumentSourceNames = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::MOGG_ARG_SRC_NAMES);
    auto argsIoSpecsAttr = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::kMOGGArgsIOSpecs);

    uint64_t begOfInputArguments =
        numberOfOutputArgumentsAttr ? numberOfOutputArgumentsAttr.getInt() : 0;
    auto funcTypeGenerator = mojoCode.getFuncTypeGenerator().getBody();
    bool resultAsArgument = false;
    for (uint64_t i = 0; i < mojoCode.getNumArguments(); ++i) {
      if (funcTypeGenerator.getArgConvention(i) ==
          KGEN::ArgConvention::ByRefResult) {
        resultAsArgument = true;
        break;
      }
    }
    bool isThrow = funcTypeGenerator.isThrows();
    uint64_t numberOfArgumentsRelatedToByrefResult =
        isThrow ? 2 : resultAsArgument;
    size_t endOfInputArguments =
        argumentTypesNames.size() - numberOfArgumentsRelatedToByrefResult;

    for (size_t i = begOfInputArguments; i < endOfInputArguments; ++i) {
      auto argTypeName = dyn_cast<StringAttr>(argumentTypesNames[i]);
      inputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                  argsIoSpecsAttr, begOfInputArguments);
    }

    for (size_t i = 0; i < begOfInputArguments; ++i) {
      auto argTypeName = dyn_cast<StringAttr>(argumentTypesNames[i]);
      // Providing no mutable tensor positions because outputs can't be mutable
      // inputs.
      outputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                   argsIoSpecsAttr);
    }

    if (resultAsArgument) {
      auto argTypeName = dyn_cast<StringAttr>(
          argumentTypesNames[endOfInputArguments +
                             numberOfArgumentsRelatedToByrefResult - 1]);
      if (argTypeName) {
        outputResult = MojoKernelOperandAdaptor(
            endOfInputArguments + numberOfArgumentsRelatedToByrefResult - 1,
            argTypeName, argumentSourceNames, argsIoSpecsAttr,
            /*isByRefResult=*/true);
      }
    } else if (resultTypeNameAttr) {
      auto resultTypeName = dyn_cast<StringAttr>(resultTypeNameAttr);
      // Providing no mutable or fused tensor positions because they don't make
      // sense for output results.
      outputResult = MojoKernelOperandAdaptor(
          {}, resultTypeName, argumentSourceNames, argsIoSpecsAttr);
    }
  }

  // Synthesize the execute function from an elementwise method.
  static MojoKernelFunctionAdaptor
  synthesizeExecuteFromElementwise(KGEN::LIT::FnOp op) {
    KGEN::LIT::FnOp mojoCode = op;
    MojoKernelFunctionAdaptor res;
    res.mojoCode = op;
    auto argumentTypesNames = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::MOGG_ARG_TYPE_NAMES);
    auto numberOfOutputArgumentsAttr =
        mojoCode->template getAttrOfType<IntegerAttr>(
            KGEN::MOGGPreElab::kMOGGNumDPSOutputs);
    auto argumentSourceNames = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::MOGG_ARG_SRC_NAMES);
    auto argsIoSpecsAttr = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::kMOGGArgsIOSpecs);

    uint64_t begOfInputArguments =
        numberOfOutputArgumentsAttr ? numberOfOutputArgumentsAttr.getInt() : 0;
    auto funcTypeGenerator = mojoCode.getFuncTypeGenerator().getBody();
    bool resultAsArgument = false;
    for (uint64_t i = 0; i < mojoCode.getNumArguments(); ++i) {
      if (funcTypeGenerator.getArgConvention(i) ==
          KGEN::ArgConvention::ByRefResult) {
        resultAsArgument = true;
        break;
      }
    }
    bool isThrow = funcTypeGenerator.isThrows();
    uint64_t numberOfArgumentsRelatedToByrefResult =
        isThrow ? 2 : resultAsArgument;
    size_t endOfInputArguments =
        argumentTypesNames.size() - numberOfArgumentsRelatedToByrefResult;

    for (size_t i = begOfInputArguments; i < endOfInputArguments; ++i) {
      auto argTypeName = dyn_cast<StringAttr>(argumentTypesNames[i]);
      if (i + 1 == endOfInputArguments &&
          argTypeName ==
              KGEN::MOGGPreElab::MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_NAME) {
        // The last argument of an elementwise function can be an IndexList
        // argument. Skip it for the execute function.
        continue;
      }
      res.inputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                      argsIoSpecsAttr, begOfInputArguments,
                                      /*promoteSIMDToFusedTensor=*/true);
    }

    // Manually build the output tensor for the elementwise op.
    res.outputArguments.push_back(
        MojoKernelOperandAdaptor::buildElementwiseOutputOperand());
    return res;
  }

  inline bool isView() {
    return mojoCode->hasAttr(KGEN::MOGGPreElab::kMOGGViewKernel);
  }

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    StringRef sourceName;
    if (mojoCode.getSymName().has_value())
      sourceName = *mojoCode.getSymName();

    if (auto name = mojoCode->getAttr("sourceName"))
      sourceName = cast<StringAttr>(name).strref();

    os << nesting << sourceName.str() << "\n" << nesting << "(\n";

    for (auto &arg : outputArguments)
      arg.printNested(os, nesting + "  [out] ") << ",\n";

    for (auto &arg : inputArguments)
      arg.printNested(os, nesting + "  [in]  ") << ",\n";

    os << nesting << ")";
    if (outputResult.has_value())
      os << " -> " << *outputResult;

    return os;
  }

private:
  MojoKernelFunctionAdaptor() = default;
};

// A helper struct that provides useful information about a kernel after being
// lowered through the Mojo compiler.
struct MojoKernelAdaptor {
  StringRef originalStructName;
  // The name of the op it implements.
  StringRef registeredOpName;
  // The kernel's entry point.
  std::optional<MojoKernelFunctionAdaptor> executeFunction;
  // The kernel's optional shape function.
  std::optional<MojoKernelFunctionAdaptor> shapeFunction;

  // The kernel's optional elementwise function.
  std::optional<MojoKernelFunctionAdaptor> elementwiseFunction;

  MojoKernelAdaptor(StringRef structName, KGEN::LIT::FnOp execute,
                    KGEN::LIT::FnOp shape, KGEN::LIT::FnOp elementwise)
      : originalStructName(structName) {
    if (execute) {
      executeFunction.emplace(execute);
      registeredOpName = execute
                             ->template getAttrOfType<StringAttr>(
                                 KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel)
                             .strref();
    }

    if (shape) {
      shapeFunction.emplace(shape);
      registeredOpName = shape
                             ->template getAttrOfType<StringAttr>(
                                 KGEN::MOGGPreElab::kMOGGShapeFunctionLabel)
                             .strref();
    }

    if (elementwise) {
      elementwiseFunction.emplace(elementwise);
      registeredOpName =
          elementwise
              ->template getAttrOfType<StringAttr>(
                  KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel)
              .strref();
      // TODO: We should always synthetize the execute signature. (GEX-2453).
      if (!execute) {
        // We synthetize the execute function signature from the elementwise
        // function.
        executeFunction =
            MojoKernelFunctionAdaptor::synthesizeExecuteFromElementwise(
                elementwise);
      }
    }
  }

  /// TODO: (GEX-1994) Remove this.
  bool isCoreMOOperation() const { return registeredOpName.starts_with("mo."); }

  bool isElementwise() const { return elementwiseFunction.has_value(); }
};

//===----------------------------------------------------------------------===//
// Printers
//===----------------------------------------------------------------------===//

template <typename StreamType>
StreamType &operator<<(StreamType &os, const MojoKernelAdaptor &adaptor) {
  os << "@register(" << adaptor.registeredOpName.str() << ")\n";
  os << "struct " << adaptor.originalStructName.str();
  if (adaptor.executeFunction.has_value()) {
    os << "\n";
    adaptor.executeFunction->printNested(os, "  ");
  }

  if (adaptor.shapeFunction.has_value()) {
    os << "\n";
    adaptor.shapeFunction->printNested(os, "  ");
  }

  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const MojoKernelFunctionAdaptor &func) {
  return func.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const MojoKernelOperandAdaptor &operand) {
  return operand.printNested(os, "");

  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const MojoKernelOperandVariant &var) {
  std::visit([&](auto &&arg) { os << arg; }, var);

  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const MojoKernelOperandSourceDescriptor &desc) {
  return os << desc.sourceName.str() << ": ";
}

static inline const char *bool2str(bool val) { return val ? "True" : "False"; }

template <typename StreamType>
StreamType &operator<<(StreamType &os, const TensorOperandAdaptor &tensor) {
  os << "Tensor[mut=" << bool2str(tensor.mut)
     << ", fused=" << bool2str(tensor.fused) << "]";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const VariadicTensorOperandAdaptor &tensors) {
  os << "*" << *static_cast<const TensorOperandAdaptor *>(&tensors);
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const ScalarOperandAdaptor &) {
  os << "Scalar";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const DevicesContextPtrOperandAdaptor &) {
  os << "Context";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const DevicesContextPtrListOperandAdaptor &) {
  os << "*Context";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const OpaqueOperandAdaptor &opaque) {
  os << "<" << opaque.typeName.str() << ">";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const UnsupportedOperandAdaptor &opaque) {
  os << "Unsupported";
  return os;
}

} // namespace M::KGEN::MOGGPreElab

#endif // MOJO_KERNEL_ADAPTOR_H
