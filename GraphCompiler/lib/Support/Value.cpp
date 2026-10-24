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

#include "GraphCompiler/Support/Value.h"
#include "GraphCompiler/Support/Dict.h"
#include "GraphCompiler/Support/List.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/LLVMForwardDecls.h"
#include <sstream>

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace GML;

None &None::operator=(None &&other) { return *this; }

bool None::operator==(const None &other) const { return true; }

bool None::operator<(const None &other) const { return false; }

std::string None::to_string() const { return "<none>"; }

raw_ostream &M::GML::operator<<(raw_ostream &os, const None &none) {
  os << none.to_string();
  return os;
}

std::ostream &M::GML::operator<<(std::ostream &os, const None &none) {
  os << none.to_string();
  return os;
}

raw_ostream &M::GML::operator<<(raw_ostream &os, const DenseMapInfoKey &info) {
  os << std::to_string(info);
  return os;
}
std::ostream &M::GML::operator<<(std::ostream &os,
                                 const DenseMapInfoKey &info) {
  os << std::to_string(info);
  return os;
}

Variant::Variant() : info(DenseMapInfoKey::Empty), value() {}

Variant::Variant(const AsyncRT::AnyAsyncValueRef &other)
    : info(DenseMapInfoKey::Empty) {
  if (other.isValueAvailable()) {
    info = DenseMapInfoKey::Value;
    value = other.copy();
  }
}

Variant::Variant(const Variant &other) : info(other.info) {
  if (info == DenseMapInfoKey::Value && other.value.isValueAvailable())
    value = other.getValue().copy();
}

Variant::Variant(const DenseMapInfoKey &info)
    : info(info), value(AsyncRT::AnyAsyncValueRef()) {}

std::string Variant::to_string() const { return valueToString(value); }

DenseMapInfoKey Variant::getInfo() const { return info; }

const AsyncRT::AnyAsyncValueRef &Variant::getValue() const { return value; }

Variant &Variant::operator=(const Variant &other) {
  info = other.info;
  if (info == DenseMapInfoKey::Value && other.value.isValueAvailable())
    value = other.getValue().copy();
  return *this;
}

Variant &Variant::operator=(Variant &&other) {
  info = other.info;
  if (info == DenseMapInfoKey::Value && other.value.isValueAvailable()) {
    value = std::move(other.value);
  }
  return *this;
}

bool M::GML::operator==(const Variant &lhs, const Variant &rhs) {
  if (lhs.getInfo() != rhs.getInfo() ||
      lhs.getInfo() == DenseMapInfoKey::Tombstone ||
      rhs.getInfo() == DenseMapInfoKey::Tombstone)
    return false;
  else if (lhs.getInfo() == DenseMapInfoKey::Value)
    return valueEqual(lhs.getValue(), rhs.getValue());
  else
    return true;
}

bool M::GML::operator!=(const Variant &lhs, const Variant &rhs) {
  return !(lhs == rhs);
}

bool M::GML::operator<(const Variant &lhs, const Variant &rhs) {
  if (lhs.getInfo() < rhs.getInfo())
    return true;
  else if (lhs.getInfo() > rhs.getInfo())
    return false;
  else if (lhs.getInfo() == DenseMapInfoKey::Value)
    return valueLessThan(lhs.getValue(), rhs.getValue());
  else
    return false;
}

raw_ostream &M::GML::operator<<(raw_ostream &os, const Variant &variant) {
  os << variant.to_string();
  return os;
}

std::ostream &M::GML::operator<<(std::ostream &os, const Variant &variant) {
  os << variant.to_string();
  return os;
}

ErrorOrSuccess M::GML::printGenericValue(raw_ostream &os,
                                         const AsyncRT::AnyAsyncValueRef &value,
                                         ResultOutputStyle style,
                                         unsigned precision) {
  if (List *list = dyn_cast<List>(value)) {
    size_t size = list->getElements().size();
    os << "List[";
    for (auto [index, value] : llvm::enumerate(list->getElements())) {
      ErrorOrSuccess errorOr =
          M::GML::printGenericValue(os, value.getValue(), style, precision);
      if (errorOr.isError())
        return errorOr;
      if (index < size - 1)
        os << ", ";
    }
    os << "]";
  } else if (Tuple *tuple = dyn_cast<Tuple>(value)) {
    size_t size = tuple->getElements().size();
    os << "Tuple[";
    for (auto [index, value] : llvm::enumerate(tuple->getElements())) {
      ErrorOrSuccess errorOr =
          M::GML::printGenericValue(os, value.getValue(), style, precision);
      if (errorOr.isError())
        return errorOr;
      if (index < size - 1)
        os << ", ";
    }
    os << "]";
  } else if (Dict *dict = dyn_cast<Dict>(value)) {
    size_t size = dict->getElements().size();
    os << "Dict{";
    for (auto [index, value] : llvm::enumerate(dict->getElements())) {
      os << value.first << ": ";
      ErrorOrSuccess errorOr = M::GML::printGenericValue(
          os, value.second.getValue(), style, precision);
      if (errorOr.isError())
        return errorOr;
      if (index < size - 1)
        os << ", ";
    }
    os << "}";
  } else if (None *none = dyn_cast<None>(value)) {
    os << none;
  } else if (Tensor *tensor = dyn_cast<Tensor>(value)) {
    ErrorOrSuccess errorOr = printTensor(os, tensor->getBuffer(),
                                         tensor->getSpec(), style, precision);
    if (errorOr.isError())
      return errorOr;
  } else {
    os << valueToString(value);
  }
  return success();
}

std::string M::GML::valueToString(const AsyncRT::AnyAsyncValueRef &value) {
  std::stringstream ss;
  if (List *list = dyn_cast<List>(value))
    ss << *list;
  else if (Tuple *tuple = dyn_cast<Tuple>(value))
    ss << *tuple;
  else if (Dict *dict = dyn_cast<Dict>(value))
    ss << *dict;
  else if (None *none = dyn_cast<None>(value))
    ss << *none;
  else if (std::string *stringValue = dyn_cast<std::string>(value))
    ss << "\"" << *stringValue << "\"";
  else if (double *doubleValue = dyn_cast<double>(value))
    ss << *doubleValue;
  else if (int64_t *longValue = dyn_cast<int64_t>(value))
    ss << *longValue;
  else if (bool *boolValue = dyn_cast<bool>(value))
    ss << *boolValue;
  else if (Tensor *tensor = dyn_cast<Tensor>(value))
    ss << tensor->getBuffer();
  else if (int32_t *intValue = dyn_cast<int32_t>(value))
    ss << *intValue;
  else
    return "<unknown>";
  return ss.str();
}

bool M::GML::valueEqual(const AsyncRT::AnyAsyncValueRef &lhs,
                        const AsyncRT::AnyAsyncValueRef &rhs) {
  if (List *llist = dyn_cast<List>(lhs)) {
    if (List *rlist = dyn_cast<List>(rhs))
      return *llist == *rlist;
  } else if (Tuple *ltuple = dyn_cast<Tuple>(lhs)) {
    if (Tuple *rtuple = dyn_cast<Tuple>(rhs))
      return *ltuple == *rtuple;
  } else if (Dict *ldict = dyn_cast<Dict>(lhs)) {
    if (Dict *rdict = dyn_cast<Dict>(rhs))
      return *ldict == *rdict;
  } else if (None *lnone = dyn_cast<None>(lhs)) {
    if (None *rnone = dyn_cast<None>(rhs))
      return *lnone == *rnone;
  } else if (std::string *lstring = dyn_cast<std::string>(lhs)) {
    if (std::string *rstring = dyn_cast<std::string>(rhs))
      return *lstring == *rstring;
  } else if (double *ldouble = dyn_cast<double>(lhs)) {
    if (double *rdouble = dyn_cast<double>(rhs))
      return *ldouble == *rdouble;
  } else if (int64_t *llong = dyn_cast<int64_t>(lhs)) {
    if (int64_t *rlong = dyn_cast<int64_t>(rhs))
      return *llong == *rlong;
  } else if (bool *lbool = dyn_cast<bool>(lhs)) {
    if (bool *rbool = dyn_cast<bool>(rhs))
      return *lbool == *rbool;
  } else if (Tensor *ltensor = dyn_cast<Tensor>(lhs)) {
    if (Tensor *rtensor = dyn_cast<Tensor>(rhs))
      // Consider tensors equal if the underlying data pointer is the same.
      return ltensor->getBuffer() == rtensor->getBuffer();
  }
  return false;
}

bool M::GML::valueLessThan(const AsyncRT::AnyAsyncValueRef &lhs,
                           const AsyncRT::AnyAsyncValueRef &rhs) {
  if (List *llist = dyn_cast<List>(lhs)) {
    if (List *rlist = dyn_cast<List>(rhs))
      return *llist < *rlist;
  } else if (Tuple *ltuple = dyn_cast<Tuple>(lhs)) {
    if (Tuple *rtuple = dyn_cast<Tuple>(rhs))
      return *ltuple < *rtuple;
  } else if (Dict *ldict = dyn_cast<Dict>(lhs)) {
    if (Dict *rdict = dyn_cast<Dict>(rhs))
      return *ldict < *rdict;
  } else if (None *lnone = dyn_cast<None>(lhs)) {
    if (None *rnone = dyn_cast<None>(rhs))
      return *lnone < *rnone;
  } else if (std::string *lstring = dyn_cast<std::string>(lhs)) {
    if (std::string *rstring = dyn_cast<std::string>(rhs))
      return *lstring < *rstring;
  } else if (double *ldouble = dyn_cast<double>(lhs)) {
    if (double *rdouble = dyn_cast<double>(rhs))
      return *ldouble < *rdouble;
  } else if (int64_t *llong = dyn_cast<int64_t>(lhs)) {
    if (int64_t *rlong = dyn_cast<int64_t>(rhs))
      return *llong < *rlong;
  } else if (bool *lbool = dyn_cast<bool>(lhs)) {
    if (bool *rbool = dyn_cast<bool>(rhs))
      return *lbool < *rbool;
  } else if (Tensor *ltensor = dyn_cast<Tensor>(lhs)) {
    if (Tensor *rtensor = dyn_cast<Tensor>(rhs))
      // Consider tensors equal if the underlying data pointer is the same.
      return ltensor->getBuffer() < rtensor->getBuffer();
  }
  return false;
}

size_t M::GML::valueHash(const AsyncRT::AnyAsyncValueRef &value) {
  if (List *list = dyn_cast<List>(value))
    return std::hash<ListLike>()(*list);
  else if (Tuple *tuple = dyn_cast<Tuple>(value))
    return std::hash<ListLike>()(*tuple);
  else if (Dict *dict = dyn_cast<Dict>(value))
    return std::hash<Dict>()(*dict);
  else if (None *none = dyn_cast<None>(value))
    return std::hash<None>()(*none);
  else if (std::string *stringValue = dyn_cast<std::string>(value))
    return std::hash<std::string>()(*stringValue);
  else if (double *doubleValue = dyn_cast<double>(value))
    return std::hash<double>()(*doubleValue);
  else if (int64_t *longValue = dyn_cast<int64_t>(value))
    return std::hash<int64_t>()(*longValue);
  else if (bool *boolValue = dyn_cast<bool>(value))
    return std::hash<bool>()(*boolValue);
  else if (Tensor *tensor = dyn_cast<Tensor>(value))
    return reinterpret_cast<size_t>(tensor->getBuffer());
  else
    return std::numeric_limits<size_t>::max();
}

size_t M::GML::hashCombine(size_t seed, size_t value) {
  // 64-bit `boost::hash_combine`; the constant is 2^64 / golden ratio.
  return seed ^ (value + 0x9e3779b97f4a7c15LLU + (seed << 6) + (seed >> 2));
}

std::string std::to_string(const None &none) { return none.to_string(); }

std::string std::to_string(const DenseMapInfoKey &value) {
  switch (value) {
  case DenseMapInfoKey::Value:
    return "<set>";
  case DenseMapInfoKey::Empty:
    return "<empty>";
  case DenseMapInfoKey::Tombstone:
    return "<tombstone>";
  }
  llvm_unreachable("invalid Torch::DenseMapInfoKey");
}

std::string std::to_string(const GML::Variant &variant) {
  return valueToString(variant.getValue());
}

size_t std::hash<GML::Variant>::operator()(const GML::Variant &variant) const {
  return hashCombine(static_cast<size_t>(variant.getInfo()),
                     valueHash(variant.getValue()));
}
