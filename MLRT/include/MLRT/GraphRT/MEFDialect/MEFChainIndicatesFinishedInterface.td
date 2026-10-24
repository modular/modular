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

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFCHAININDICATESFINISHED_TD
#define MLRT_GRAPHRT_MEFDIALECT_MEFCHAININDICATESFINISHED_TD

include "mlir/IR/OpBase.td"

def ChainIndicatesFinishedInterface : OpInterface<"ChainIndicatesFinishedInterface"> {
  let cppNamespace = "::M::MEF";

  let description = [{
    This interface indicates that the op is FINISHED execution when its
    `!mef.chain` results are ready, even though the ops other results may not
    yet be finished. The op must have at least one `!mef.chain` result. Here
    'FINISHED' means the op no longer holds any references to its operands,
    and any sub-tasks started by the op have completed.

    Ops without this interface are assumed to follow a stricter rule and be
    FINISHED when ANY of their results are ready. This stricter rule applies
    irrespective of whether the op is marked as `NonStrictInterface`.

    This interface is not used at runtime, only compile time (by dataflow
    analysis).

    See also NonStrictInterface, which describes how op operands interact with
    op execution.
  }];
}

#endif // MLRT_GRAPHRT_MEFDIALECT_MEFCHAININDICATESFINISHED_TD
