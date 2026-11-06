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
// Platforms of kernel registrations.
//
// A kernel registration applies to a set of devices, given as a KGEN
// generator: its four parameters are the label, API, architecture, and model
// of a device, and its body combines equality tests on those fields with
// `and` and `or`. The Mojo side is `extensibility/platform_predicates.mojo`.
//
// `PlatformPredicate` holds the generator as a small expression. It evaluates
// the expression on a device, and it compares two expressions exactly: each
// field is compared only with string literals, so all strings that the
// compared expressions do not mention behave the same. Thus a finite set of
// representative devices covers all devices.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PLATFORMPREDICATE_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PLATFORMPREDICATE_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace M::MOGG {

/// The device fields that a platform reads, in the order of the parameters of
/// a platform generator.
enum class DeviceField : unsigned { Label = 0, Api, Arch, Model };
inline constexpr unsigned kNumDeviceFields = 4;

/// The fields of a concrete device, in `DeviceField` order.
using DeviceFields = std::array<llvm::StringRef, kNumDeviceFields>;

/// Returns the fields of `device` that platforms read.
DeviceFields getDeviceFields(DeviceInfoAttr device);

/// A literal for each field, in `DeviceField` order.
using FieldLiterals =
    std::array<llvm::SmallVector<std::string>, kNumDeviceFields>;

/// A representative device for the comparison of platforms. A field without
/// a value stands for every string that is not in `excluded` for that field.
/// The compared platforms cannot tell those strings apart.
struct SymbolicDevice {
  std::array<std::optional<std::string>, kNumDeviceFields> values;
  FieldLiterals excluded;

  /// Prints the device, for example
  /// `type="gpu", api="cuda", arch not in ["sm_100a"], model=*`.
  void print(llvm::raw_ostream &os) const;
};

/// The set of devices that a kernel registration applies to.
class PlatformPredicate {
public:
  /// Returns a platform that holds for all devices.
  PlatformPredicate();

  /// Reads the canonical expression of a platform generator. Returns an error
  /// if the expression uses anything other than `and`, `or`, boolean
  /// constants, and equality of device fields to string literals.
  static ErrorOr<PlatformPredicate>
  fromGenerator(KGEN::GeneratorAttr generator);

  /// Returns the platform that a `device_info` registration selects: each
  /// non-empty field must equal its value.
  // TODO(GEX-4475): delete with `registration_device_info`, once every
  // registration states a platform.
  static PlatformPredicate fromDeviceInfo(DeviceInfoAttr device);

  bool holdsFor(const DeviceFields &device) const;
  bool holdsFor(const SymbolicDevice &device) const;

  /// Adds each literal that the platform compares a field with to the list
  /// of that field. The lists can contain duplicates.
  void collectLiterals(FieldLiterals &literals) const;

  /// Prints the platform in the syntax of `extensibility`, for example
  /// `type("gpu") & (api("cuda") | api("hip"))`.
  void print(llvm::raw_ostream &os) const;
  std::string str() const;

private:
  struct Node {
    enum class Kind { Constant, Equals, And, Or };
    Kind kind = Kind::Constant;
    bool value = true;          // Constant.
    DeviceField field{};        // Equals.
    std::string literal;        // Equals.
    std::vector<Node> operands; // And, Or.
  };

  explicit PlatformPredicate(Node root) : root(std::move(root)) {}

  static ErrorOr<Node> parse(mlir::TypedAttr expr);
  template <typename FieldValue>
  static bool evaluate(const Node &node, FieldValue fieldValue);
  static void collectLiterals(const Node &node, FieldLiterals &literals);
  static void print(const Node &node, llvm::raw_ostream &os, bool nested);

  Node root;
};

/// How the sets of devices of two platforms relate.
enum class PlatformPredicateOrder {
  Equivalent,     // The sets are equal.
  FirstNarrower,  // The first set is a strict subset of the second.
  SecondNarrower, // The second set is a strict subset of the first.
  Incomparable,   // Each set has a device that the other does not have.
};

/// The result of `comparePlatformPredicates`, with an example device for each
/// difference and for the overlap, when one exists.
struct PlatformPredicateComparison {
  PlatformPredicateOrder order = PlatformPredicateOrder::Equivalent;
  std::optional<SymbolicDevice> firstOnly;
  std::optional<SymbolicDevice> secondOnly;
  std::optional<SymbolicDevice> both;
};

/// Compares the sets of devices of two platforms exactly.
PlatformPredicateComparison
comparePlatformPredicates(const PlatformPredicate &first,
                          const PlatformPredicate &second);

/// Calls `fn` on each representative device of `platforms`: each field takes
/// each literal that a platform compares it with, or a value that no
/// platform mentions.
void forEachRepresentativeDevice(
    llvm::ArrayRef<const PlatformPredicate *> platforms,
    llvm::function_ref<void(const SymbolicDevice &)> fn);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PLATFORMPREDICATE_H
