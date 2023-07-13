//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_JSON_H
#define SUPPORT_JSON_H

namespace llvm {
namespace json {
class Value;
}
class raw_ostream;
} // namespace llvm

namespace M {

/// Serialize the provided JSON value in its canonical form as described by
/// RFC8785. This function will recurse on itself for any sub-objects inside
/// `v`. There is one non-standard detail - and that is that the values in the
/// JSON are expected to be UTF-8 encoded, not UTF-16 encoded.
void serializeCanonicalJSON(const llvm::json::Value *v, llvm::raw_ostream &os);

} // namespace M

#endif // SUPPORT_JSON_H
