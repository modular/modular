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

#include "GraphCompiler/MOGGDialect/Support/PlatformPredicate.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace M;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// The `extensibility` atom that tests each field.
constexpr llvm::StringLiteral kFieldAtomNames[kNumDeviceFields] = {
    "type", "api", "arch", "model"};

std::string printAttr(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << attr;
  return text;
}

std::string printType(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  return text;
}

TypedAttr stripWrappers(TypedAttr attr) {
  return KGEN::ParamOperatorAttr::stripRebind(KGEN::SugarAttr::strip(attr));
}

/// Returns the device field that `term` reads, or nothing if `term` is not
/// one of the generator's parameters.
std::optional<DeviceField> getFieldOfTerm(TypedAttr term) {
  auto ref = dyn_cast<KGEN::ParamIndexRefAttr>(stripWrappers(term));
  if (!ref || ref.getDepth() != 0 || ref.getIndex() >= kNumDeviceFields)
    return std::nullopt;
  return static_cast<DeviceField>(ref.getIndex());
}

void sortAndRemoveDuplicates(SmallVectorImpl<std::string> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

} // namespace

DeviceFields M::MOGG::getDeviceFields(DeviceInfoAttr device) {
  return {device.getLabel(), device.getApi(), device.getArch(),
          device.getModel()};
}

void SymbolicDevice::print(llvm::raw_ostream &os) const {
  for (unsigned i = 0; i < kNumDeviceFields; ++i) {
    if (i)
      os << ", ";
    os << kFieldAtomNames[i];
    if (values[i]) {
      os << "=\"" << *values[i] << "\"";
      continue;
    }
    if (excluded[i].empty()) {
      os << "=*";
      continue;
    }
    os << " not in [";
    llvm::interleave(
        excluded[i], os, [&](const std::string &s) { os << '"' << s << '"'; },
        ", ");
    os << "]";
  }
}

PlatformPredicate::PlatformPredicate() = default;

ErrorOr<PlatformPredicate>
PlatformPredicate::fromGenerator(KGEN::GeneratorAttr generator) {
  // `rebind` in Mojo can give a generator of any type to `register`, so check
  // that it has the type of `extensibility.PlatformExpr`.
  ArrayRef<Type> inputs = generator.getInputParamTypes();
  if (inputs.size() != kNumDeviceFields ||
      !llvm::all_of(inputs,
                    [](Type type) { return isa<KGEN::StringType>(type); }) ||
      !KGEN::isScalarOf<KGEN::KGENDType::kBool>(generator.getBody().getType()))
    return Error("a platform must take the label, api, arch, and "
                 "model of a device as strings and return a bool, not " +
                 printType(generator.getType()));

  ErrorOr<Node> root = parse(
      KGEN::SugarAttr::strip(KGEN::getCanonicalAttr(generator.getBody())));
  if (root.isError())
    return Error(root.getError());
  return PlatformPredicate(root.takeValue());
}

PlatformPredicate PlatformPredicate::fromDeviceInfo(DeviceInfoAttr device) {
  Node conjunction;
  conjunction.kind = Node::Kind::And;
  DeviceFields fields = getDeviceFields(device);
  for (unsigned i = 0; i < kNumDeviceFields; ++i) {
    if (fields[i].empty())
      continue;
    Node equals;
    equals.kind = Node::Kind::Equals;
    equals.field = static_cast<DeviceField>(i);
    equals.literal = fields[i].str();
    conjunction.operands.push_back(std::move(equals));
  }
  if (conjunction.operands.empty())
    return PlatformPredicate();
  if (conjunction.operands.size() == 1)
    return PlatformPredicate(std::move(conjunction.operands.front()));
  return PlatformPredicate(std::move(conjunction));
}

ErrorOr<PlatformPredicate::Node> PlatformPredicate::parse(TypedAttr expr) {
  expr = KGEN::SugarAttr::strip(expr);

  if (KGEN::isTriviallyTrueProposition(expr) ||
      KGEN::isTriviallyFalseProposition(expr)) {
    Node constant;
    constant.value = KGEN::isTriviallyTrueProposition(expr);
    return constant;
  }

  if (auto op = dyn_cast<KGEN::ParamOperatorAttr>(expr)) {
    if (op.getOpcode() == KGEN::POC::And || op.getOpcode() == KGEN::POC::Or) {
      Node node;
      node.kind =
          op.getOpcode() == KGEN::POC::And ? Node::Kind::And : Node::Kind::Or;
      for (TypedAttr operand : op.getOperands()) {
        ErrorOr<Node> child = parse(operand);
        if (child.isError())
          return Error(child.getError());
        node.operands.push_back(child.takeValue());
      }
      return node;
    }
  }

  // KGEN merges `api == "x" and arch == "x"` into one identity class,
  // `identical(api, arch, "x")`, so an identity can name several fields. Each
  // class keeps at least one literal, because every atom compares a field
  // with a literal.
  if (auto identical = dyn_cast<KGEN::ParamIdenticalAttr>(expr)) {
    SmallVector<DeviceField> fields;
    SmallVector<std::string> literals;
    for (TypedAttr operand : identical.getOperands()) {
      if (std::optional<DeviceField> field = getFieldOfTerm(operand)) {
        if (!llvm::is_contained(fields, *field))
          fields.push_back(*field);
        continue;
      }
      if (auto literal =
              dyn_cast<StringAttr>(KGEN::SugarAttr::strip(operand))) {
        literals.push_back(literal.getValue().str());
        continue;
      }
      return Error("unsupported operand in platform: " + printAttr(operand));
    }
    sortAndRemoveDuplicates(literals);
    if (fields.empty() || literals.empty())
      return Error("a platform must compare each device field with a "
                   "string literal, not " +
                   printAttr(expr));
    // A field cannot be equal to two different strings.
    if (literals.size() > 1) {
      Node constant;
      constant.value = false;
      return constant;
    }
    Node conjunction;
    conjunction.kind = Node::Kind::And;
    for (DeviceField field : fields) {
      Node equals;
      equals.kind = Node::Kind::Equals;
      equals.field = field;
      equals.literal = literals.front();
      conjunction.operands.push_back(std::move(equals));
    }
    if (conjunction.operands.size() == 1)
      return std::move(conjunction.operands.front());
    return conjunction;
  }

  return Error("unsupported expression in platform: " + printAttr(expr));
}

template <typename FieldValue>
bool PlatformPredicate::evaluate(const Node &node, FieldValue fieldValue) {
  switch (node.kind) {
  case Node::Kind::Constant:
    return node.value;
  case Node::Kind::Equals:
    return fieldValue(node.field, node.literal);
  case Node::Kind::And:
    return llvm::all_of(node.operands, [&](const Node &operand) {
      return evaluate(operand, fieldValue);
    });
  case Node::Kind::Or:
    return llvm::any_of(node.operands, [&](const Node &operand) {
      return evaluate(operand, fieldValue);
    });
  }
  llvm_unreachable("unknown platform node");
}

bool PlatformPredicate::holdsFor(const DeviceFields &device) const {
  return evaluate(root, [&](DeviceField field, llvm::StringRef literal) {
    return device[static_cast<unsigned>(field)] == literal;
  });
}

bool PlatformPredicate::holdsFor(const SymbolicDevice &device) const {
  return evaluate(root, [&](DeviceField field, llvm::StringRef literal) {
    const std::optional<std::string> &value =
        device.values[static_cast<unsigned>(field)];
    return value && *value == literal;
  });
}

void PlatformPredicate::collectLiterals(const Node &node,
                                        FieldLiterals &literals) {
  if (node.kind == Node::Kind::Equals)
    literals[static_cast<unsigned>(node.field)].push_back(node.literal);
  for (const Node &operand : node.operands)
    collectLiterals(operand, literals);
}

void PlatformPredicate::collectLiterals(FieldLiterals &literals) const {
  collectLiterals(root, literals);
}

void PlatformPredicate::print(const Node &node, llvm::raw_ostream &os,
                              bool nested) {
  switch (node.kind) {
  case Node::Kind::Constant:
    os << (node.value ? "true" : "false");
    return;
  case Node::Kind::Equals:
    os << kFieldAtomNames[static_cast<unsigned>(node.field)] << "(\""
       << node.literal << "\")";
    return;
  case Node::Kind::And:
  case Node::Kind::Or:
    if (nested)
      os << "(";
    llvm::interleave(
        node.operands, os,
        [&](const Node &operand) { print(operand, os, /*nested=*/true); },
        node.kind == Node::Kind::And ? " & " : " | ");
    if (nested)
      os << ")";
    return;
  }
}

void PlatformPredicate::print(llvm::raw_ostream &os) const {
  print(root, os, /*nested=*/false);
}

std::string PlatformPredicate::str() const {
  std::string text;
  llvm::raw_string_ostream os(text);
  print(os);
  return text;
}

void M::MOGG::forEachRepresentativeDevice(
    llvm::ArrayRef<const PlatformPredicate *> platforms,
    llvm::function_ref<void(const SymbolicDevice &)> fn) {
  FieldLiterals domains;
  for (const PlatformPredicate *platform : platforms)
    platform->collectLiterals(domains);
  for (SmallVector<std::string> &domain : domains)
    sortAndRemoveDuplicates(domain);

  // Index `domain.size()` of a field stands for a string that no platform
  // mentions.
  std::array<size_t, kNumDeviceFields> choice{};
  SymbolicDevice device;
  while (true) {
    for (unsigned i = 0; i < kNumDeviceFields; ++i) {
      if (choice[i] < domains[i].size()) {
        device.values[i] = domains[i][choice[i]];
        device.excluded[i].clear();
      } else {
        device.values[i] = std::nullopt;
        device.excluded[i] = domains[i];
      }
    }
    fn(device);

    unsigned i = 0;
    for (; i < kNumDeviceFields; ++i) {
      if (++choice[i] <= domains[i].size())
        break;
      choice[i] = 0;
    }
    if (i == kNumDeviceFields)
      return;
  }
}

PlatformPredicateComparison
M::MOGG::comparePlatformPredicates(const PlatformPredicate &first,
                                   const PlatformPredicate &second) {
  PlatformPredicateComparison result;
  const PlatformPredicate *platforms[] = {&first, &second};
  forEachRepresentativeDevice(platforms, [&](const SymbolicDevice &d) {
    bool inFirst = first.holdsFor(d);
    bool inSecond = second.holdsFor(d);
    if (inFirst && !inSecond && !result.firstOnly)
      result.firstOnly = d;
    if (inSecond && !inFirst && !result.secondOnly)
      result.secondOnly = d;
    if (inFirst && inSecond && !result.both)
      result.both = d;
  });

  if (result.firstOnly && result.secondOnly)
    result.order = PlatformPredicateOrder::Incomparable;
  else if (result.firstOnly)
    result.order = PlatformPredicateOrder::SecondNarrower;
  else if (result.secondOnly)
    result.order = PlatformPredicateOrder::FirstNarrower;
  else
    result.order = PlatformPredicateOrder::Equivalent;
  return result;
}
