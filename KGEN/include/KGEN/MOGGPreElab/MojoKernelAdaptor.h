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

struct ListOfTensorOperandAdaptor : TensorOperandAdaptor {
  static constexpr StringLiteral typeName = MOJO_TENSOR_LIST_NAME;
  bool operator==(const ListOfTensorOperandAdaptor &other) const {
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

struct MojoKernelOperandSourceDescriptor {
  // The name of the variable in the original Mojo code.
  StringRef sourceName;
  // Its position in the function.
  uint64_t position;
};

using MojoKernelOperandVariant =
    std::variant<TensorOperandAdaptor, VariadicTensorOperandAdaptor,
                 ListOfTensorOperandAdaptor, ScalarOperandAdaptor,
                 OpaqueOperandAdaptor, DevicesContextPtrOperandAdaptor,
                 DevicesContextPtrListOperandAdaptor>;

struct MojoKernelOperandAdaptor {
  // The source information which is optional in the case of a return value.
  std::optional<MojoKernelOperandSourceDescriptor> sourceDescriptor;
  // A union between all kinds of supported operands.
  MojoKernelOperandVariant underlyingType;

  MojoKernelOperandAdaptor(std::optional<uint64_t> positionInFunction,
                           StringRef typeName, ArrayAttr argumentSourceNames,
                           ArrayAttr argsIoSpecs, uint64_t offset = 0);

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

  bool isVariadicTensorType() const {
    return std::holds_alternative<VariadicTensorOperandAdaptor>(underlyingType);
  }

  bool isOpaqueType() const {
    return std::holds_alternative<OpaqueOperandAdaptor>(underlyingType);
  }

  bool isContextType() const {
    return std::holds_alternative<DevicesContextPtrOperandAdaptor>(
               underlyingType) ||
           std::holds_alternative<DevicesContextPtrListOperandAdaptor>(
               underlyingType);
  }

  bool isTensorType() const {
    return std::holds_alternative<TensorOperandAdaptor>(underlyingType) ||
           std::holds_alternative<VariadicTensorOperandAdaptor>(
               underlyingType) ||
           std::holds_alternative<ListOfTensorOperandAdaptor>(underlyingType);
  }
};

template <typename FuncOpType>
struct MojoKernelFunctionAdaptor {
  // The underlying LIT or KGEN function.
  // Marked mutable because getSymName is not const for some reason.
  mutable FuncOpType mojoCode;
  // Input arguments.
  SmallVector<MojoKernelOperandAdaptor> inputArguments;
  // Output arguments (DPS).
  SmallVector<MojoKernelOperandAdaptor> outputArguments;
  // Output result which could be an argument if the function throws.
  std::optional<MojoKernelOperandAdaptor> outputResult;

  MojoKernelFunctionAdaptor(FuncOpType op) : mojoCode(op) {
    auto argumentTypesNames = mojoCode->template getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::MOGG_ARG_TYPE_NAMES);
    auto resultTypeName = mojoCode->template getAttrOfType<StringAttr>(
        KGEN::MOGGPreElab::MOGG_RESULT_TYPE_NAME);
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
      auto argTypeName = cast<StringAttr>(argumentTypesNames[i]).strref();
      inputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                  argsIoSpecsAttr, begOfInputArguments);
    }

    for (size_t i = 0; i < begOfInputArguments; ++i) {
      auto argTypeName = cast<StringAttr>(argumentTypesNames[i]).strref();
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
        // Providing no mutable or fused tensor positions because they don't
        // make sense for output results.
        outputResult = MojoKernelOperandAdaptor(
            endOfInputArguments + numberOfArgumentsRelatedToByrefResult - 1,
            argTypeName.strref(), argumentSourceNames, argsIoSpecsAttr);
      }
    } else if (resultTypeName) {
      // Providing no mutable or fused tensor positions because they don't make
      // sense for output results.
      outputResult = MojoKernelOperandAdaptor(
          {}, resultTypeName.strref(), argumentSourceNames, argsIoSpecsAttr);
    }
  }

  bool isElementwiseFunction() {
    return mojoCode->hasAttr(KGEN::MOGGPreElab::kMOGGElementwiseLambda);
  }

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    StringRef sourceName;
    if constexpr (std::is_same_v<LIT::FnOp, FuncOpType>) {
      if (mojoCode.getSymName().has_value())
        sourceName = *mojoCode.getSymName();
    }

    if (auto name = mojoCode->getAttr("sourceName"))
      sourceName = cast<StringAttr>(name).strref();
    os << nesting << sourceName.str() << "\n" << nesting << "(\n";

    for (auto &arg : outputArguments)
      arg.printNested(os, nesting + "  [out] ") << ",\n";

    for (auto arg : inputArguments)
      arg.printNested(os, nesting + "  [in]  ") << ",\n";

    os << nesting << ")";
    if (outputResult.has_value())
      os << " -> " << *outputResult;

    return os;
  }
};

// A helper struct that provides useful information about a kernel after being
// lowered through the Mojo compiler.
template <typename FuncOpType>
struct MojoKernelAdaptor {
  StringRef originalStructName;
  // The name of the op it implements.
  StringRef registeredOpName;
  // The kernel's entry point.
  std::optional<MojoKernelFunctionAdaptor<FuncOpType>> executeFunction;
  // The kernel's optional shape function.
  std::optional<MojoKernelFunctionAdaptor<FuncOpType>> shapeFunction;

  MojoKernelAdaptor(StringRef structName, FuncOpType execute, FuncOpType shape)
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
  }

  /// TODO: (GEX-1994) Remove this.
  bool isCoreMOOperation() const { return registeredOpName.starts_with("mo."); }
};

//===----------------------------------------------------------------------===//
// Printers
//===----------------------------------------------------------------------===//

template <typename StreamType, typename FuncOpType>
StreamType &operator<<(StreamType &os,
                       const MojoKernelAdaptor<FuncOpType> &adaptor) {
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

template <typename StreamType, typename FuncOpType>
StreamType &operator<<(StreamType &os,
                       const MojoKernelFunctionAdaptor<FuncOpType> &func) {
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
  os << "*" << *reinterpret_cast<const TensorOperandAdaptor *>(&tensors);
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const ListOfTensorOperandAdaptor &list) {
  os << "List[" << *reinterpret_cast<const TensorOperandAdaptor *>(&list)
     << "]";
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

} // namespace M::KGEN::MOGGPreElab

#endif // MOJO_KERNEL_ADAPTOR_H
