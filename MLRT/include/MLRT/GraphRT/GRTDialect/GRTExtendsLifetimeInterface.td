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

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD

include "mlir/IR/OpBase.td"

def GRTExtendsLifetimeInterface : OpInterface<"GRTExtendsLifetimeInterface"> {
  let description = [{
    This interface indicates an operation may extend the lifetime of its
    operands to include the lifetime of its results. That is, a result
    may point to an AsyncValue which holds a pointer to an operand AsyncValue.

    However, !mef.chain operands and results are not subject to this rule.

    The AsyncDataflowDependencies analysis requires this information so as
    to soundly approximate the 'exclusive' relation between SSA values.
  }];

  let cppNamespace = "::M::GRT";

  let methods = [
    InterfaceMethod<
      [{
        Returns true if operand with index operandIndex has its lifetime
        extended to include that of the result with resultIndex.
      }],
      "bool", "extendsLifetime", (ins "size_t": $operandIndex, "size_t": $resultIndex)
    >,
  ];

}

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD
