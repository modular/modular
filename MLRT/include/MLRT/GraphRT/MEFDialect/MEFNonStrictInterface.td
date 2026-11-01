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

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFNONSTRICTINTERFACE_TD
#define MLRT_GRAPHRT_MEFDIALECT_MEFNONSTRICTINTERFACE_TD

include "mlir/IR/OpBase.td"

def NonStrictInterface : OpInterface<"NonStrictInterface"> {
  let cppNamespace = "::M::MEF";

  let description = [{
    This interface indicates to the MEF execution engine that the op may be
    scheduled to BEGIN execution before all of its operands are ready. The op
    itself then takes responsibility for correctly sequencing computation as its
    arguments become ready.

    Ops without this interface are only scheduled once all operands are ready.
    There's nothing the op implementation needs to do in this case.

    This interface is used at both runtime (by the MEF executor) and compile
    time (by dataflow analysis).

    See also ChainIndicatesFinishedInterface, which describes how op result
    connect with op execution.
  }];
}

#endif // MLRT_GRAPHRT_MEFDIALECT_MEFNONSTRICTINTERFACE_TD
