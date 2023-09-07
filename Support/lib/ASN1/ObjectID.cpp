//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ASN1/ObjectID.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace ASN1;

//===----------------------------------------------------------------------===//
// Variable Length Quantity (VLQ)
//===----------------------------------------------------------------------===//

/// Count the number of octets needed for a VLQ integer. The number of octets is
/// the ceiling div of the number of bytes in the integer with any non-zero bits
/// in them. Each octet in a VLQ only contains 7 bits, because the top bit is
/// used to indicate if it's the last octet.
static size_t vlqNumOctets(uint64_t val) {
  size_t numRequiredBits = ((sizeof(val) * 8) - llvm::countl_zero(val | 1));
  size_t numOctets = (numRequiredBits + 7 - 1) / 7; // ceiling division
  return numOctets;
}

/// Encode a number as a VLQ. The long form is only needed if `value` is > 0x7f
/// - anything less and it's a short form VLQ (just the value itself).
static void encodeVLQ(uint64_t value, SmallVectorImpl<uint8_t> &buf) {
  if (value <= 0x7f) {
    buf.push_back((uint8_t)value);
    return;
  }

  // Use the long form.
  size_t numOctets = vlqNumOctets(value);
  uint64_t mask = 0x7full << ((numOctets - 1) * 7);
  do {
    uint8_t val = (value & mask) >> ((numOctets - 1) * 7);
    // The high bit is set for octets that are not the last.
    buf.push_back(val | 0x80);
    mask >>= 7;
    --numOctets;
  } while (numOctets > 1);
  // Last octet.
  uint8_t val = value & mask;
  buf.push_back(val);
}

/// Decode a VLQ from `buf`. The VLQ could technically be much larger than a
/// uint64_t, but since we don't deal with those quantities here, we can make
/// this simplifying assumption.
static ErrorOr<uint64_t> decodeVLQ(ArrayRef<uint8_t> buf,
                                   size_t *bytesConsumed) {
  if (buf.empty())
    return Error("cannot decode a VLQ from an empty buffer");

  if (buf.front() < 0x7f) {
    *bytesConsumed = 1;
    return (uint64_t)buf.front();
  }

  if (buf.front() == 0x80) {
    return Error("invalid encoding of an object identifier, the leading octet "
                 "must not be 0x80");
  }

  // Otherwise, decode the bytes one at a time.
  uint64_t out = 0;
  const uint8_t *bufPtr = buf.begin();
  const uint8_t *bufEnd = buf.end();
  // Save the last byte we pulled out of the buffer. This is used to check for
  // error cases later.
  uint8_t lastByte = 0;
  // We can consume at most 9 bytes into a uint64_t.
  for (int i = 0; i < 9 && (bufPtr != bufEnd); ++i) {
    // Take the front byte and increment bufPtr.
    lastByte = *bufPtr++;
    // Shift `out` and put the value into out.
    out <<= 7;
    out |= lastByte & (~0x80);
    // If this value doesn't have the continuation bit set, then we're done.
    if (!(lastByte & 0x80))
      break;
  }

  // If the last byte has the continuation bit set, then either the stream is
  // truncated or the number was too large.
  if (lastByte & 0x80) {
    if (bufPtr == buf.end())
      return Error("truncated stream, last byte had the continuation bit set");

    return Error(
        "byte stream contains a VLQ too large to encode in a single uint64_t");
  }

  // bufPtr always points to one past the last byte we processed, so the number
  // of bytes processed is just a pointer subtraction.
  *bytesConsumed = bufPtr - buf.begin();

  return out;
}

//===----------------------------------------------------------------------===//
// ObjectID
//===----------------------------------------------------------------------===//

ObjectID::ObjectID(bool withModularPrefix, ArrayRef<uint64_t> arc)
    : withModularPrefix(withModularPrefix) {
  // Canonicalize the stored arc - if it starts with modularPrefixNums then we
  // don't have to store the modular prefix. Otherwise, store the entire arc. We
  // have to make sure we don't expect the modular prefix here because otherwise
  // it'll get re-canonicalized on copy or move.
  if (!withModularPrefix && arc.take_front(modularPrefix.size()) ==
                                ArrayRef<uint64_t>(modularPrefix)) {
    llvm::append_range(this->arc, arc.drop_front(modularPrefix.size()));
    this->withModularPrefix = true;
  } else {
    llvm::append_range(this->arc, arc);
  }
}

void ObjectID::append(ArrayRef<uint64_t> nodes) {
  arc.append(nodes.begin(), nodes.end());
}

SmallVector<uint8_t> ObjectID::getEncoded() const {
  SmallVector<uint8_t> out;
  // If we're using the modular prefix, we already have it encoded correctly and
  // in byte form. Otherwise, we have to encode it the way ASN.1 expects the
  // first 2 nodes of an OID to be encoded.
  if (isModularOID())
    llvm::append_range(out, getEncodedModularPrefix());
  else
    out.push_back((uint8_t)getArc()[0] * 40 + (uint8_t)getArc()[1]);

  // If we don't have the modular prefix, then we can drop the first two
  // elements in the arc.
  ArrayRef toEncode = getArc();
  if (!isModularOID())
    toEncode = toEncode.drop_front(2);

  // The way OID nodes are encoded is as VLQs.
  for (auto val : toEncode)
    encodeVLQ(val, out);

  return out;
}

ErrorOr<ObjectID> ObjectID::fromEncoded(ArrayRef<uint8_t> buf) {
  // If we start with the modularPrefix, then drop it.
  bool hasModularPrefix = buf.take_front(encodedModularPrefix.size()) ==
                          ArrayRef<uint8_t>(encodedModularPrefix);
  if (hasModularPrefix)
    buf = buf.drop_front(encodedModularPrefix.size());

  ObjectID out(hasModularPrefix);
  // If we don't have the modular prefix, then the first two nodes are encoded
  // as first * 40 + second.
  if (!hasModularPrefix) {
    uint8_t first = buf.front();
    buf = buf.drop_front();

    out.append({(uint64_t)first / 40, (uint64_t)first % 40});
  }

  // While we have buffer left, decode the VLQs.
  while (!buf.empty()) {
    size_t bytesConsumed = 0;
    auto numOr = decodeVLQ(buf, &bytesConsumed);
    if (numOr.isError())
      return numOr.takeError();

    // Add the number we just parsed, and drop the bytes that were consumed.
    out.append(*numOr);
    buf = buf.drop_front(bytesConsumed);
  }

  return out;
}

ErrorOr<ObjectID> ObjectID::fromString(StringRef str) {
  SmallVector<StringRef> nums;
  str.split(nums, '.');
  if (llvm::any_of(nums, [](StringRef num) { return num.empty(); }))
    return Error("unexpected empty number in oid " + str);

  SmallVector<uint64_t> oid;
  for (StringRef num : nums)
    if (num.getAsInteger(10, oid.emplace_back()))
      return Error("found non-integer in oid: " + num);

  // Check if we have the modular prefix - if we do, then we can drop the prefix
  // and store just the namespaced arc. If not, we have to store the whole
  // thing.
  bool withModularPrefix =
      ArrayRef(oid).take_front(modularPrefix.size()) == ArrayRef(modularPrefix);
  ObjectID out(withModularPrefix);
  if (!withModularPrefix)
    out.append(oid);
  else
    out.append(ArrayRef(oid).drop_front(modularPrefix.size()));

  return out;
}

/// Two ObjectID objects are equal if their arcs are equal. Since the Modular
/// prefix is constant, we only have to check their arcs.
bool ASN1::operator==(const ObjectID &lhs, const ObjectID &rhs) {
  return lhs.getArc() == rhs.getArc() &&
         lhs.isModularOID() == rhs.isModularOID();
}

llvm::raw_ostream &M::operator<<(llvm::raw_ostream &os,
                                 const ASN1::ObjectID &oid) {
  if (oid.isModularOID()) {
    llvm::interleave(
        ASN1::ObjectID::getModularPrefix(), os,
        [&](uint8_t u) { os << (int)u; }, ".");

    // Separator dot between prefix and the arc.
    os << ".";
  }

  // Print the arc.
  llvm::interleave(oid.getArc(), os, ".");

  return os;
}
