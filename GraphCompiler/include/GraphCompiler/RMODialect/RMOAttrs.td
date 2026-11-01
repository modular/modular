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
// This defines attributes in the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_TD

include "GraphCompiler/RMODialect/RMODialect.td"
include "Support/Compiler/AttrBase.td"
include "mlir/IR/AttrTypeBase.td"

class RMOAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RMODialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_TD
