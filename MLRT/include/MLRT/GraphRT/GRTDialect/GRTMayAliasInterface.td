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

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD

include "mlir/IR/OpBase.td"

def GRTMayAliasInterface : OpInterface<"GRTMayAliasInterface"> {
  let description = [{
    This interface indicates an operation may return aliases to its operands.
    Here 'alias' includes:
      - 'direct' aliasing: a result may be a pointer to an operand AsyncValue.
         For example, the identity op.
      - 'internal' aliasing: a result is a pointer to a fresh AsyncValue which
         shares an underlying resource with an operand AsyncValue. For example,
         an op to create a tensor from a buffer which may move the buffer if
         it has no other references.

    The AsyncDataflowDependencies analysis requires this information so as
    to soundly approximate the 'exclusive' relation between SSA values.
  }];

  let cppNamespace = "::M::GRT";

  let methods = [
    InterfaceMethod<
      [{
        Returns true if operand with index operandIndex may alias with result
        resultIndex.
      }],
      "bool", "mayAlias", (ins "size_t": $operandIndex, "size_t": $resultIndex)
    >,
  ];
}

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD
