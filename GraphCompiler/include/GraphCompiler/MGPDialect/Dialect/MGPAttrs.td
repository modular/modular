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
// This defines attributes in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPATTRS_TD
#define GENERICML_MGPDIALECT_MGPATTRS_TD

include "GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class MGP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MGPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// StableAddressArrayAttr
//===----------------------------------------------------------------------===//

def MGP_StableAddressArrayAttr
    : ArrayOfAttr<MGPDialect, "StableAddressArray", "stable_addr_array",
                  "::M::MGP::StableAddressType"> {
  let summary = "The stable address slots a device graph collective declares.";
  let description = [{
    An ordered list of `!mgp.stable_addr` types.
  }];
}

// Currently no attributes.

#endif  // GENERICML_MGPDIALECT_MGPATTRS_TD
