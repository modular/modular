//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENCALLINTERFACE_TD
#define KGEN_KGENDIALECT_KGENCALLINTERFACE_TD

include "mlir/IR/OpBase.td"

def KGEN_CallOpInterface : OpInterface<"KGENCallOpInterface"> {
  let description = [{
    This interface represents a call to a parametric KGEN declaration: either a
    function, generator, or interface.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the callee expression of the call operation.
    }], "TypedAttr", "getCallee">,

    InterfaceMethod<[{
      Get the signature type of the callee.
    }], "SignatureType", "getSignature", (ins), [{}], [{
      return cast<SignatureType>($_op.getCallee().getType());
    }]>,
    InterfaceMethod<[{
      Get the function type of the callee.
    }], "FunctionType", "getFunctionType", (ins), [{}], [{
      return $_op.getSignature().getValues();
    }]>,

    InterfaceMethod<[{
      Get the parameters declared by the call.
    }], "ParamDeclArrayAttr", "getParamDeclsAttr">,

    InterfaceMethod<[{
      Get the bound region bodies of this operation.
    }], "iterator_range<CallRegionBodyIterator>", "getRegionBodies", (ins),
    [{}], [{
      return llvm::map_range($_op->getRegions(), +[](Region &region) {
        return cast<DeclInterface>(region.front().front());
      });
    }]>,
    InterfaceMethod<[{
      Returns whether the call operation's regions are isolated from above.
    }], "bool", "hasIsolatedRegions", (ins), [{}], [{ return true; }]>
  ];
}

#endif // KGEN_KGENDIALECT_KGENCALLINTERFACE_TD
