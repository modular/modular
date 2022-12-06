//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENCALLINTERFACE_TD
#define KGEN_KGENDIALECT_KGENCALLINTERFACE_TD

include "mlir/IR/OpBase.td"

def KGENCallOpInterface : OpInterface<"KGENCallOpInterface"> {
  let description = [{
    This interface represents a call to a parametric KGEN declaration: either a
    function, generator, or interface.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the function type of the callee.
    }], "FunctionType", "getFunctionType", (ins), [{}], [{
      return FunctionType::get($_op->getContext(), $_op->getOperandTypes(),
                               $_op->getResultTypes());
    }]>,

    InterfaceMethod<[{
      Get the parameters declared by the call.
    }], "ParamDeclArrayAttr", "getParamDeclsAttr", (ins), [{}]>,

    InterfaceMethod<[{
      Get the FnEffects and parameters conventions of the callee.
    }], "DenseI8ArrayAttr", "getConventionsAttr", (ins), [{}]>,

    InterfaceMethod<[{
      Get the FnEffects and parameters conventions of the callee.
    }], "ArrayRef<int8_t>", "getConventions", (ins), [{}]>,

    InterfaceMethod<[{
      Get the bound region bodies of this operation.
    }], "iterator_range<CallRegionBodyIterator>", "getRegionBodies", (ins),
    [{}], [{
      return llvm::map_range($_op->getRegions(), +[](Region &region) {
        return cast<KGENDeclInterface>(region.front().front());
      });
    }]>,
    InterfaceMethod<[{
      Returns whether the call operation's regions are isolated from above.
    }], "bool", "hasIsolatedRegions", (ins), [{}], [{ return true; }]>
  ];
}

#endif // KGEN_KGENDIALECT_KGENCALLINTERFACE_TD
