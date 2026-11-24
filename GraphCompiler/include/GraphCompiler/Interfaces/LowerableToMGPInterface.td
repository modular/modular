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

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMGPINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMGPINTERFACE_TD

include "mlir/IR/OpBase.td"

def LowerableToMGPInterface :
    OpInterface<"LowerableToMGPInterface"> {
  let description = [{
    This interface provides a way for dialects to implement methods needed to
    lower themselves during the MOGGToMGP pass.
  }];

  let cppNamespace = "::M::MGP";

  let methods = [
    InterfaceMethod<
      [{
        Lower this op into primitives compatible with MGP. Implementers make
        use of the shared lowering state object, and mutate it appropriately. In
        particular, they need to:
         * use the builders returned by getExecuteOpBuilder() and
           getInitOpBuilder() to insert the lowered ops into their appropriate
           location
         * update the lowering state using emplaceExecuteLoweredValue(...) for
           any new outputs they generate, and use get{Execute,Init}Chain() as
           needed
      }],
      "mlir::LogicalResult",
      "lower",
      (ins "M::MGP::MOGraphLoweringState &":$lowering)>,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMGPINTERFACE_TD
