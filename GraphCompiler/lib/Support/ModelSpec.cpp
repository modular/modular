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
// This file implements an in-memory representation for a model specification.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Support/ModelSpec.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
using namespace M;

//===----------------------------------------------------------------------===//
// ModelSpec::NameSubstitutions
//===----------------------------------------------------------------------===//
StringRef
ModelSpec::NameSubstitutions::getPreferredName(StringRef preferredName) const {
  auto itr = preferredToNativeMap.find(preferredName);
  return itr == preferredToNativeMap.end() ? preferredName : itr->second;
}

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

ModelSpec::~ModelSpec() = default;

ErrorOr<const ModelSpec::ModelBinary *>
ModelSpec::getModelBinary(StringRef format) const {
  auto itr = std::find_if(model.begin(), model.end(),
                          [format](const ModelBinary &modelBinary) {
                            return modelBinary.format == format;
                          });
  if (itr == model.end())
    return Error(
        Twine("The model spec does not contain an entry for format '") +
        format + "'");
  return &*itr;
}
