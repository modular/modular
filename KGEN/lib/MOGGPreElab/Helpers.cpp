//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Helpers.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"

namespace M::KGEN::MOGGPreElab {

// Returns true if all inputs are extensibility tensors.
static bool isExtensibilityKernel(GeneratorOp generator) {
  std::optional<LIT::LITSignatureType> litSig = getSourceSig(generator);
  if (!litSig.has_value())
    return false;

  for (Type metadata : litSig->getValues().getInputs()) {
    // Tensors are expected to be passed as references.
    auto asLitRef = dyn_cast<LIT::RefType>(metadata);
    if (!asLitRef)
      return false;

    auto asDeclRef =
        dyn_cast<KGEN::LIT::DeclRefType>(asLitRef.getElementType());
    if (!asDeclRef)
      return false;

    if (!isExtensibilityTensor(asDeclRef))
      return false;
  }

  return true;
}

static void annotateExtensibilityKernels(GeneratorOp func,
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

    newAttrs.push_back(
        NamedAttribute{b.getStringAttr("_alloc"), b.getIndexArrayAttr(allocs)});
  }
}

bool stripDecorators(GeneratorOp func) {
  SmallVector<TypedAttr> decoratorsToCopy;
  OpBuilder builder{func.getContext()};

  bool areAnyKernels = false;

  // We will replace each decorator with a new attribute.
  SmallVector<NamedAttribute> newAttrs;

  // Decorators we should replace with a trivial unit attribute.
  constexpr std::array<StringLiteral, 11> identityDecorators{
      DECORATOR_ELEM_HOOK,
      DECORATOR_ELEMENTWISE,
      DECORATOR_ELEMENTWISE_PUBLIC,
      DECORATOR_VIEW,
      DECORATOR_TAKES_INDICES,
      DECORATOR_TENSOR_ALLOC,
      DECORATOR_TENSOR_COPY_CONSTRUCT,
      DECORATOR_TENSOR_DECONSTRUCT,
      DECORATOR_ENABLE_FUSION_HOOK,
      DECORATOR_INPUT_FUSION_HOOK,
      DECORATOR_OUTPUT_FUSION_HOOK};

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
    StringRef decoratorName;

    // The decorator might just be a direct symbol, for instance `@decorator`
    // vs `@decorator()`.
    auto directSym = dyn_cast<SymbolConstantAttr>(decorator);
    if (directSym)
      decoratorName = directSym.getSymbol().getLeafReference().strref();

    // We track the apply so we can pull extra arguments from it.
    // `@decorator("Arg1", 100)`
    ParamOperatorAttr apply;

    // Otherwise the other allowed form of decorators are parameter apply
    // expressions of the symbol. E.G `@decorator()`
    if (decoratorName.empty()) {
      apply = dyn_cast<ParamOperatorAttr>(decorator);
      if (apply) {
        // The first operand is expected to be the symbol we are applying.
        if (auto sym = dyn_cast<SymbolConstantAttr>(apply.getOperand(0)))
          decoratorName = sym.getSymbol().getLeafReference().strref();
      }
    }

    if (decoratorName.empty())
      continue;

    for (StringLiteral target : identityDecorators) {
      if (decoratorName.starts_with(target)) {
        newAttrs.push_back(NamedAttribute{builder.getStringAttr(target),
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
    if (decoratorName.starts_with(DECORATOR_REGISTER_OVERRIDE) ||
        decoratorName.starts_with(DECORATOR_REGISTER_PUBLIC_OVERRIDE)) {
      // Register kernels with explicit override.
      kernelRegistrations.push_back(cast<StringAttr>(apply.getOperand(1)));
      kernelRegistrations.push_back(cast<IntegerAttr>(apply.getOperand(2)));
      decoratorsToCopy.pop_back();
      areAnyKernels = true;
    } else if (decoratorName.starts_with(DECORATOR_REGISTER_SHAPE_FUNC)) {
      // Register V1 shape functions.
      shapeFunctionReg.push_back(cast<StringAttr>(apply.getOperand(1)));
      decoratorsToCopy.pop_back();
      areAnyKernels = true;
    } else if (decoratorName.starts_with(DECORATOR_REGISTER_KERNEL)) {
      // Register kernels without explict override parameter.
      kernelRegistrations.push_back(cast<StringAttr>(apply.getOperand(1)));
      kernelRegistrations.push_back(builder.getI64IntegerAttr(-1));
      decoratorsToCopy.pop_back();
      areAnyKernels = true;
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
