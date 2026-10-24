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
//
// This contains the GraphRTDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTDIALECT_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTDIALECT_TD

include "mlir/IR/DialectBase.td"

def GRTDialect : Dialect {
  let name = "grt";

  let summary =
    "The 'grt' dialect holds definitions of widely used primitive ops.";
  let description = [{
    This is the `grt` dialect, which holds commonly used primitives for
    working with integers and implementing control flow.
  }];

  let cppNamespace = "::M::GRT";

  // We depend on the MEF dialect for the chain type.
  let dependentDialects = ["::M::MEF::MEFDialect"];
}

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTDIALECT_TD
