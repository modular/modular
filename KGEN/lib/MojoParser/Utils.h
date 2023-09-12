//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#include <cstddef>

namespace M::KGEN {
class SignatureType;
} // namespace M::KGEN

namespace M::KGEN::POP {
class PackType;
} // namespace M::KGEN::POP

namespace M::KGEN::LIT {
/// If the argument at the given index is of pack type, returns that type.
/// therwise, returns null.
POP::PackType getIfPackType(SignatureType sig, size_t index);
} // namespace M::KGEN::LIT
