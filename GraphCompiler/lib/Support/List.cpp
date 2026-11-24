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

#include "GraphCompiler/Support/List.h"
#include "GraphCompiler/Support/Dict.h"
#include "GraphCompiler/Support/Value.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <sstream>
#include <utility>

using namespace M;
using namespace M::GML;

ListLike::ListLike(const SmallVector<Variant> &elements) : elements(elements) {}

ListLike::ListLike(const ListLike &other) {
  for (const Variant &elem : other.getElements())
    getElements().push_back(
        AsyncRT::AnyAsyncValueRef::copy(elem.getValue().getPointer()));
}

ListLike::ListLike(ListLike &&other) : elements(std::move(other.elements)) {}

ListLike &ListLike::operator=(ListLike &&other) {
  elements.clear();
  elements = std::move(other.elements);
  return *this;
}

SmallVector<Variant> &ListLike::getElements() { return elements; }
const SmallVector<Variant> &ListLike::getElements() const { return elements; }

raw_ostream &M::GML::operator<<(raw_ostream &os, const ListLike &list) {
  os << list.to_string();
  return os;
}

std::ostream &M::GML::operator<<(std::ostream &os, const ListLike &list) {
  os << list.to_string();
  return os;
}

Tuple Tuple::operator=(const Tuple &other) {
  elements.clear();
  std::transform(other.elements.begin(), other.elements.end(),
                 this->elements.begin(),
                 [](const Variant &value) { return value.getValue().copy(); });
  return *this;
}

Tuple &Tuple::operator=(Tuple &&other) {
  elements.clear();
  elements = std::move(other.elements);
  return *this;
}

std::string Tuple::to_string() const {
  std::stringstream ss;
  ss << "Tuple(";
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    ss << elements[i] << (i == e - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

bool M::GML::operator==(const Tuple &lhs, const Tuple &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] != rhs.getElements()[i])
      return false;
  }
  return true;
}

bool M::GML::operator!=(const Tuple &lhs, const Tuple &rhs) {
  return !(lhs == rhs);
}

bool M::GML::operator<(const Tuple &lhs, const Tuple &rhs) {
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] < rhs.getElements()[i])
      return true;
  }
  if (lhs.getElements().size() < rhs.getElements().size())
    return true;
  return false;
}

List::List(ArrayRef<AnyAsyncValueRef> values) {
  for (auto &value : values)
    elements.emplace_back(std::move(value));
}

List List::operator=(const List &other) {
  elements.clear();
  std::transform(other.elements.begin(), other.elements.end(),
                 this->elements.begin(),
                 [](const Variant &value) { return value.getValue().copy(); });
  return *this;
}

List &List::operator=(List &&other) {
  elements.clear();
  elements = std::move(other.elements);
  return *this;
}

List List::copy() {
  List listCopy;
  for (const Variant &variant : elements)
    listCopy.elements.push_back(variant.getValue().copy());
  return listCopy;
}

std::string List::to_string() const {
  std::stringstream ss;
  ss << "List[";
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    ss << elements[i] << (i == e - 1 ? "" : ", ");
  }
  ss << "]";
  return ss.str();
}

bool M::GML::operator==(const List &lhs, const List &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] != rhs.getElements()[i])
      return false;
  }
  return true;
}

bool M::GML::operator!=(const List &lhs, const List &rhs) {
  return !(lhs == rhs);
}

bool M::GML::operator<(const List &lhs, const List &rhs) {
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] < rhs.getElements()[i])
      return true;
  }
  if (lhs.getElements().size() < rhs.getElements().size())
    return true;
  return false;
}

namespace std {

std::string to_string(const List &list) { return list.to_string(); }

std::string to_string(const Tuple &tuple) { return tuple.to_string(); }

size_t hash<ListLike>::operator()(const ListLike &list) const {
  std::size_t seed = list.getElements().size();
  for (auto &elem : list.getElements()) {
    seed ^= hashCombine(seed, std::hash<Variant>()(elem));
  }
  return seed;
}

} // namespace std
