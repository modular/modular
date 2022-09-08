//===- ElaboratorOpInterface.td -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_ELABORATOROPINTERFACE_TD
#define KGEN_KGENDIALECT_ELABORATOROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ElaboratorOpInterface : OpInterface<"ElaboratorOpInterface"> {
  let description = [{
    The KGEN elaborator interface defines a set of methods that allow operations
    to interact with the KGEN elaborator. The KGEN elaborator interacts with
    generic operations by inspecting their types and attributes for parameters
    to substitute, using `SubElementAttrInterface` and
    `SubElementTypeInterface`. This interface allows operations to hook into
    specific elaborator stages beyond generic parameter substitution.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      This method is invoked on the operation to indicate that its parametric
      attributes and types have been fully elaborated. The operation should, if
      necessary, internally reify itself given its fully-elaborated attributes
      and types.

      This is useful for operations that have looser semantics pre-elaboration
      but stronger guarantees post-elaboration and which require custom logic
      to switch between those modes.

      This method is invoked after the operation has been elaborated but before
      any nested operations are elaborated.
    }], "ErrorOrSuccess", "finalizeElaboration", (ins), [{}], [{
      return success();
    }]>,
  ];
}

#endif // KGEN_KGENDIALECT_ELABORATOROPINTERFACE_TD
