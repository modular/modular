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

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMOGGINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMOGGINTERFACE_TD

include "mlir/IR/OpBase.td"

def LowerableToMOGGInterface :
    OpInterface<"LowerableToMOGGInterface"> {
  let description = [{
    This interface provides a way for `map` dialect ops to implement their own
    lowering into `mogg` codegen ops during the MAPToMOGG pass.
  }];

  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      [{
        Lower this op into mogg ops, using `state`'s builder and reading/
        recording value substitutions through `state.mapping`. A region-owning
        op recurses into its own body via `state.lowerRegionBody`.
      }],
      "mlir::LogicalResult",
      "lowerToMOGG",
      (ins "M::MAP::MAPToMOGGState &":$state)>,
    InterfaceMethod<
      [{
        Lower this op inside a tile fusion body, whose values are whole
        `!mogg.tile`s rather than SIMD vectors. Only ops that can appear in
        such a body override it.
      }],
      "mlir::LogicalResult",
      "lowerToMOGGTile",
      (ins "M::MAP::MAPToMOGGState &":$state),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->emitOpError("MAPToMOGG: op has no tile lowering");
      }]>,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMOGGINTERFACE_TD
