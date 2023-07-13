//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/JSON.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <codecvt>
#include <locale>

using namespace M;
using namespace llvm;

void M::serializeCanonicalJSON(const json::Value *v, raw_ostream &os) {
  // Nothing there, write "null" to the stream (see
  // https://www.rfc-editor.org/rfc/rfc8785#name-serialization-of-literals)
  if (!v) {
    os << "null";
    return;
  }

  // Not an array or object, simply write it to the stream.
  if (v->kind() != json::Value::Array && v->kind() != json::Value::Object) {
    os << *v;
    return;
  }

  // It's an array - maintain sorted order within the array but any subobjects
  // must be themselves properly sorted.
  if (const json::Array *arr = v->getAsArray()) {
    os << "[";
    interleave(
        *arr, [&](const json::Value &val) { serializeCanonicalJSON(&val, os); },
        [&]() { os << ","; });
    os << "]";
  }

  // It's an object, so we have to sort it.
  const json::Object *obj = v->getAsObject();
  assert(obj && "there's nothing else it could be, it must be an object");
  os << "{";

  // Sort the keys (we want stable sort here so that it's always deterministic).
  SmallVector<json::ObjectKey> keys = to_vector(make_first_range(*obj));
  // This uses operator< on llvm::json::ObjectKey, which calls operator< of
  // StringRef. The only goal is a deterministic property order, so we can
  // really use any sort as long as it's the same between here and the producer.
  stable_sort(keys);

  // For each of the keys, write the key itself and the object at that key.
  interleave(
      keys,
      [&](const json::ObjectKey &k) {
        os << "\"" << k << "\":";
        serializeCanonicalJSON(obj->get(k), os);
      },
      [&]() { os << ","; });
  os << "}";
}
