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

#include "GraphCompiler/Support/Dict.h"
#include "GraphCompiler/Support/List.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <numeric>
#include <sstream>

using namespace M;
using namespace M::GML;

Dict::Dict(const Dict &other) {
  for (const auto &elem : other.getElements()) {
    getElements()[elem.first] = Variant(
        AsyncRT::AnyAsyncValueRef::copy(elem.second.getValue().getPointer()));
  }
}

Dict &Dict::operator=(const Dict &other) {
  elements.clear();
  for (auto &elem : other.getElements())
    elements.insert(elem);
  return *this;
}

Dict Dict::operator=(Dict &&other) {
  this->elements.clear();
  this->elements = std::move(other.elements);
  return *this;
}

llvm::MapVector<Variant, Variant> &Dict::getElements() { return elements; }

const llvm::MapVector<Variant, Variant> &Dict::getElements() const {
  return elements;
}

std::string Dict::to_string() const {
  std::stringstream ss;
  ss << "Dict{";
  size_t e = elements.size();
  for (auto [elementIndex, elementValue] : llvm::enumerate(elements)) {
    ss << elementValue.first;
    ss << ": " << elementValue.second << (elementIndex == e - 1 ? "" : ", ");
  }
  ss << "}";
  return ss.str();
}

raw_ostream &M::GML::operator<<(raw_ostream &os, const Dict &dict) {
  os << dict.to_string();
  return os;
}

std::ostream &M::GML::operator<<(std::ostream &os, const Dict &dict) {
  os << dict.to_string();
  return os;
}

bool M::GML::operator==(const Dict &lhs, const Dict &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (auto &kv : lhs.getElements()) {
    auto it = rhs.getElements().find(kv.first);
    if (it == rhs.getElements().end() || (it->second != kv.second))
      return false;
  }
  return true;
}

bool M::GML::operator!=(const Dict &lhs, const Dict &rhs) {
  return !(lhs == rhs);
}

bool M::GML::operator<(const Dict &lhs, const Dict &rhs) { return false; }

namespace std {

std::string to_string(const Dict &dict) { return dict.to_string(); }

size_t hash<Dict>::operator()(const Dict &dict) const {
  std::hash<Variant> h;
  std::size_t seed = dict.getElements().size();
  for (auto &elem : dict.getElements()) {
    seed = hashCombine(seed, h(elem.first));
    seed = hashCombine(seed, std::hash<Variant>()(elem.second));
  }
  return seed;
}

} // namespace std
