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
// Interfaces for MGP types which may also be used by other dialects which
// have similar buffer and tensor types.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPEINTERFACES_TD
#define GENERICML_MGPDIALECT_MGPTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def MGP_HasDeviceTypeInterface : TypeInterface<"MGPHasDeviceTypeInterface"> {
  let description = [{
    Types which carry an `#M.device_ref` attribute.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the type's device.", "::M::DeviceRefAttr", "getDevice", (ins)
    >
  ];
}

#endif  // GENERICML_MGPDIALECT_MGPTENSORTYPEINTERFACE_TD
