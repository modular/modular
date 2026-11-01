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
// This defines MEF dialect types, like !mef.unknown and !mef.chain.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFTYPES_TD
#define MLRT_GRAPHRT_MEFDIALECT_MEFTYPES_TD

include "MLRT/GraphRT/MEFDialect/MEFDialect.td"
include "MLRT/GraphRT/MEFDialect/MEFTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for the types in this dialect.
class MEFType<string mnemonic, list<Trait> traits = [],
              string baseCppClass = "Type"> :
    TypeDef<MEFDialect, mnemonic, traits, baseCppClass>;

//===----------------------------------------------------------------------===//
// ChainType
//===----------------------------------------------------------------------===//

def ChainType : MEFType<"Chain", [DeclareTypeInterfaceMethods<MEF_ChainTypeInterface>]> {
  let mnemonic = "chain";
  let summary = "empty value used for control dependence tracking";
  let description = [{
    This type is used to model dependencies between side-effecting operations,
    by turning these side effects into explicitly modeled values.
  }];
}

//===----------------------------------------------------------------------===//
// UnknownType
//===----------------------------------------------------------------------===//

def UnknownType : MEFType<"Unknown"> {
  let mnemonic = "unknown";
  let summary = "an unknown type";
  let description = [{
    This type is primarily used by MLIR2MEF to reflect parsed MEF files that
    don't have concrete metadata about types in the MEF file.
  }];
}

#endif // MLRT_GRAPHRT_MEFDIALECT_MEFTYPES_TD
