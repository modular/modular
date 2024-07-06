//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Helpers.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"

namespace M::KGEN::MOGGPreElab {

// The prefix that internal and max scoped decorators will start with.
constexpr StringLiteral MAX_PREFIX = "max";
constexpr StringLiteral INTERNAL_PREFIX = "register";

// Returns true if all inputs are extensibility tensors.
static bool isExtensibilityKernel(LIT::FuncOp func) {
  if (func.getNumArguments() == 0)
    return false;

  for (Type arg : func.getArgumentTypes()) {
    // Tensors are expected to be passed as references.
    auto asLitRef = dyn_cast<LIT::RefType>(arg);
    if (!asLitRef)
      return false;

    auto asDeclRef = dyn_cast<KGEN::LIT::StructType>(asLitRef.getElementType());
    if (!asDeclRef)
      return false;
    if (!isExtensibilityTensor(asDeclRef) && !isCustomType(asDeclRef))
      return false;
  }
  return true;
}

static void annotateExtensibilityKernels(LIT::FuncOp func,
                                         SmallVector<NamedAttribute> &newAttrs,
                                         OpBuilder &b) {
  // If we are a kernel and we are using the extensibility tensors we should
  // mark ourselves as allocating.
  if (isExtensibilityKernel(func)) {
    SmallVector<int64_t> allocs;

    // Mark any by ref outputs as allocating.
    for (auto [idx, convention] :
         llvm::enumerate(func.getSignature().getArgConventions())) {
      if (convention == KGEN::ArgConvention::ByRefResult)
        allocs.push_back(idx);
    }

    newAttrs.push_back(NamedAttribute{b.getStringAttr(ALLOCS_ATTR),
                                      b.getIndexArrayAttr(allocs)});
  }
}

bool stripDecorators(LIT::FuncOp func) {
  SmallVector<TypedAttr> decoratorsToCopy;
  OpBuilder builder{func.getContext()};

  bool areAnyKernels = false;

  // We will replace each decorator with a new attribute.
  SmallVector<NamedAttribute> newAttrs;

  // Decorators we should replace with a trivial unit attribute.
  constexpr std::array<MOGGDecorator, 11> identityDecorators{
      Decorators::ELEM_HOOK,
      Decorators::ELEMENTWISE,
      Decorators::VIEW,
      Decorators::TAKES_INDICES,
      Decorators::TENSOR_ALLOC,
      Decorators::TENSOR_COPY,
      Decorators::TENSOR_DECONSTRUCT,
      Decorators::ENABLE_FUSION,
      Decorators::INPUT_FUSION,
      Decorators::OUTPUT_FUSION,
      Decorators::ELEMENTWISE_PUBLIC};

  // Each kernel can implement multiple operations. We will canonicalize these
  // into one attribute.
  SmallVector<Attribute> kernelRegistrations, shapeFunctionReg;

  size_t numDecorators = func.getDecorators().size();

  // Identify which MOGG specific decorators this function has if any.
  for (TypedAttr decorator : func.getDecorators()) {
    // Keep track of the non mogg decorators to preserve them on the user
    // kernel.
    decoratorsToCopy.push_back(decorator);

    // Identify the decorator being used.
    llvm::StringRef decoratorName;

    // The decorator might just be a direct symbol, for instance `@decorator`
    // vs `@decorator()`.
    auto directSym = dyn_cast<SymbolConstantAttr>(decorator);
    if (directSym) {
      // Only accept decorators in max / register domain.
      if (!(directSym.getSymbol().getRootReference().strref() == MAX_PREFIX ||
            directSym.getSymbol().getRootReference().strref() ==
                INTERNAL_PREFIX))
        continue;
      decoratorName = directSym.getSymbol().getLeafReference().strref();
    }

    // We track the apply so we can pull extra arguments from it.
    // `@decorator("Arg1", 100)`
    ParamOperatorAttr apply;

    // Otherwise the other allowed form of decorators are parameter apply
    // expressions of the symbol. E.G `@decorator()`
    if (decoratorName.empty()) {
      apply = dyn_cast<ParamOperatorAttr>(decorator);
      if (apply) {
        // The first operand is expected to be the symbol we are applying.
        if (auto sym = dyn_cast<SymbolConstantAttr>(apply.getOperand(0))) {
          SymbolRefAttr symRef = sym.getSymbol();
          // Only accept decorators in max / register domain.
          if (!(symRef.getRootReference().strref() == "max" ||
                symRef.getRootReference().strref() == "register"))
            continue;
          decoratorName = symRef.getLeafReference().strref();
        }
      }
    }

    if (decoratorName.empty())
      continue;

    for (MOGGDecorator target : identityDecorators) {
      if (decoratorName.starts_with(target.decorator)) {
        newAttrs.push_back(NamedAttribute{builder.getStringAttr(target.attr),
                                          builder.getUnitAttr()});
        decoratorsToCopy.pop_back();
        break;
      }
    }

    // All the other decorators below are expected to be in the form of taking
    // arguments. I.E an apply expression.
    if (!apply)
      continue;

    // Kernel identifiers are slightly different as the include the name and
    // priority of the kernel.
    if (decoratorName.starts_with(Decorators::REGISTER_OVERRIDE) ||
        decoratorName.starts_with(Decorators::REGISTER_PUBLIC_OVERRIDE)) {
      // Register kernels with explicit override.
      kernelRegistrations.push_back(apply.getOperand(1));
      kernelRegistrations.push_back(apply.getOperand(2));
      decoratorsToCopy.pop_back();
      areAnyKernels = true;
    } else if (decoratorName.starts_with(Decorators::REGISTER_SHAPE_FUNC)) {
      // Register V1 shape functions.
      shapeFunctionReg.push_back(apply.getOperand(1));
      decoratorsToCopy.pop_back();
      areAnyKernels = true;
    } else if (decoratorName.starts_with(Decorators::REGISTER_KERNEL)) {
      // Register kernels without explict override parameter.
      kernelRegistrations.push_back(apply.getOperand(1));
      kernelRegistrations.push_back(builder.getI64IntegerAttr(-1));
      decoratorsToCopy.pop_back();
      areAnyKernels = true;
    } else if (decoratorName.starts_with(Decorators::REGISTER_MOGG_INTRINSIC)) {
      TypedAttr str = std::get<1>(
          cast<LIT::LITStructAttr>(apply.getOperand(1)).getValues()[0]);
      newAttrs.push_back(
          NamedAttribute{cast<StringAttr>(str), builder.getUnitAttr()});
      decoratorsToCopy.pop_back();
    }
  }

  // Implicity export all kernels and also add annotations to mark extensibility
  // types.
  if (areAnyKernels) {
    annotateExtensibilityKernels(func, newAttrs, builder);
    func.setExportKind(ExportKind::Exported);
  }

  // We don't need to do anything if we don't have any decorators.
  if (numDecorators == decoratorsToCopy.size())
    return false;

  if (!kernelRegistrations.empty()) {
    newAttrs.push_back(
        NamedAttribute{builder.getStringAttr(kernelRegistrationAttr),
                       builder.getArrayAttr(kernelRegistrations)});
  }

  if (!shapeFunctionReg.empty()) {
    newAttrs.push_back(
        NamedAttribute{builder.getStringAttr(shapeFuncRegistrationAttr),
                       builder.getArrayAttr(shapeFunctionReg)});
  }

  // Update the function to have only the non mogg decorators.
  func.setDecorators(DecoratorsAttr::get(func.getContext(), decoratorsToCopy));

  // Add all the old attributes.
  for (const NamedAttribute &attr : func->getAttrs())
    newAttrs.push_back(attr);
  func->setAttrs(newAttrs);

  return areAnyKernels;
}

} // namespace M::KGEN::MOGGPreElab
