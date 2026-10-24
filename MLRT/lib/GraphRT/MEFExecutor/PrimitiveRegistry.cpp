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

#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/AssertStream.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

//===----------------------------------------------------------------------===//
// PrimitiveRegistry::Impl
//===----------------------------------------------------------------------===//

namespace M {
struct PrimitiveRegistry::Impl {
  bool isFinalized = false;
  std::vector<std::pair<StringLiteral, PrimitiveImplFn>> prims;
};
} // namespace M

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

static void mef_model() {
  TRACY_ZONE_SCOPED_NC("mef_model", TRACY_COLOR_TEAL);

  ASSERT_STREAM(false, << "mef.model should never be invoked");
}

static void mef_callable() {}
static void builtin_module() {}

PrimitiveRegistry::PrimitiveRegistry() : impl(std::make_unique<Impl>()) {
  // FIXME: remove this and the mef_callable primitive impl above
  ADD_PRIMITIVE(*this, "mef.callable", mef_callable);
  ADD_PRIMITIVE(*this, "mef.model", mef_model);
  // FIXME: We should make it possible for others to implement this to do some
  //        substantive thing.
  ADD_PRIMITIVE(*this, "builtin.module", builtin_module);
}

PrimitiveRegistry::~PrimitiveRegistry() = default;

void PrimitiveRegistry::add(StringLiteral name, PrimitiveImplFn fn) {
  impl->prims.emplace_back(name, fn);
}

void PrimitiveRegistry::finalize() {
  // Sort by name here so we can lookup using a binary search by name later.
  llvm::sort(impl->prims, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  auto duplicate = std::adjacent_find(impl->prims.begin(), impl->prims.end());
  if (duplicate != impl->prims.end())
    llvm::report_fatal_error("Found a duplicate in the primitive registry.");

  impl->isFinalized = true;
}

auto PrimitiveRegistry::get(StringRef name) const -> PrimitiveImplFn {
  if (!impl->isFinalized)
    llvm::report_fatal_error(
        "PrimitiveRegistry must be finalized before attempting to "
        "`get` an impl from it.");

  auto it = std::lower_bound(
      impl->prims.begin(), impl->prims.end(), name,
      [](const auto &pair, StringRef name) { return pair.first < name; });

  // std::lower_bound returns the first element that is >= the searched element.
  // Therefore, we have to check if the value is equal (which is what we want).
  return it != impl->prims.end() && it->first == name ? it->second
                                                      : PrimitiveImplFn();
}

//===----------------------------------------------------------------------===//
// MEFDictionaryAttr
//===----------------------------------------------------------------------===//

/// Print the contents of this dictionary to stderr for debugging.
void MEFDictionaryAttr::dump() const {
  // This isn't doing a very full fidelity print because we don't want a bunch
  // of code size that is difficult to strip out.  Just do something simple for
  // debugging.
  fprintf(stderr, "MEFDictionaryAttr {\n");
  for (const auto &elt : *this) {
    fprintf(stderr, "  %.*s = %d\n", (int)elt.getName().size(),
            elt.getName().data(), (int)elt.getValue().getAttrKind());
  }
  fprintf(stderr, "}\n");
}
