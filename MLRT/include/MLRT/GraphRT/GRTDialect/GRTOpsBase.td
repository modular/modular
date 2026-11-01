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
// This defines the base ops in the GRT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTOPSBASE_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTOPSBASE_TD

include "MLRT/GraphRT/GRTDialect/GRTDialect.td"
include "mlir/IR/OpBase.td"

// Base class for the operation in this dialect.
class GRTOp<string mnemonic, list<Trait> traits = []> :
    Op<GRTDialect, mnemonic, traits>;

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTOPSBASE_TD
