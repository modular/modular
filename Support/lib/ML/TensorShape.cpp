//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/TensorShape.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// TensorShapeStorage
//===----------------------------------------------------------------------===//

bool Detail::TensorShapeStorage::equalsIncludingAuxOOL(
    const TensorShapeStorage &rhs) const {
  return getAuxiliary() == rhs.getAuxiliary() && equalsExcludingAuxOOL(rhs);
}

bool Detail::TensorShapeStorage::equalsExcludingAuxOOL(
    const TensorShapeStorage &rhs) const {
  return SmallVector<ssize_t, 5>(begin(), end()) ==
         SmallVector<ssize_t, 5>(rhs.begin(), rhs.end());
}

/// Bulk reassignment of elements.
/// TODO: Forcing dimensions to 64-bit is suboptimal on 32-bit hosts.
void Detail::TensorShapeStorage::assign(ArrayRef<int64_t> elements) {
  if (getRepKind() == RepKind::kOutOfLine)
    delete[] representation.repOutOfLine.dims;

  // Zero-initialize to ensure the representation value is determinsitic.
  // We do not zero out the auxiliary field.
  memset(&representation, 0, sizeof(representation) - 1);

  // Get and set the rank, regardless of the representation.
  size_t rank = elements.size();
  representation.repOutOfLine.rank = rank;
  assert(representation.repOutOfLine.rank == rank &&
         "can only handle rank up to 255");

  // Decide which representation we can use and initialize the elements.  The
  // most common case should fit into 4 dimensions.
  if (rank <= 4) {
    ssize_t dim;
    // Copy the iterator in case things don't work out.
    auto endIt = elements.end();
    switch (rank) {
    default:
      assert(0 && "unreachable");
    case 4:
      dim = *--endIt;
      representation.rep32.dim3 = dim;
      if (representation.rep32.dim3 != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 3:
      dim = *--endIt;
      representation.rep32.dims[2] = dim;
      if (representation.rep32.dims[2] != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 2:
      dim = *--endIt;
      representation.rep32.dims[1] = dim;
      if (representation.rep32.dims[1] != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 1:
      dim = *--endIt;
      representation.rep32.dims[0] = dim;
      if (representation.rep32.dims[0] != dim)
        break; // Check for dimension too large.
      [[fallthrough]];
    case 0:
      representation.rep32.kind = RepKind::k32;
      return; // Success
    }
  }

  // Virtually everything else will fit into 6 dimensions.
  if (rank <= 6) {
    size_t i;
    // Copy the iterator in case things don't work out.
    auto beginIt = elements.begin();
    for (i = 0; i < rank; ++i) {
      ssize_t dim = *beginIt++;
      representation.rep16.dims[i] = dim;
      if (representation.rep16.dims[i] != dim)
        break;
    }
    if (i == rank) {
      representation.rep16.kind = RepKind::k16;
      return; // Success
    }
  }

  // Otherwise go out of line.
  representation.repOutOfLine.kind = RepKind::kOutOfLine;
  representation.repOutOfLine.dims = new ssize_t[rank];
  std::copy(elements.begin(), elements.end(), representation.repOutOfLine.dims);
}

//===----------------------------------------------------------------------===//
// Printing, Stringizing, and Parsing
//===----------------------------------------------------------------------===//

void TensorShape::print(raw_ostream &os) const {
  llvm::interleave(
      getDims(), os,
      [&](ssize_t dim) {
        if (mlir::ShapedType::isDynamic(dim)) {
          os << "?";
          return;
        }
        os << dim;
      },
      "x");
}

std::string TensorShape::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void TensorShape::dump() const { print(llvm::errs()); }

ErrorOr<TensorShape> TensorShape::parseFromString(StringRef str) {
  // Empty strings gum up the rest of the function since splitStr would still
  // have one (empty) element, so early-out in this case.
  if (str.empty())
    return TensorShape();

  SmallVector<StringRef, 5> splitStr;
  str.split(splitStr, 'x');

  SmallVector<int64_t, 5> shape;
  shape.reserve(splitStr.size());
  for (auto &it : splitStr) {
    int64_t value;
    if (it == "?")
      value = mlir::ShapedType::kDynamic;
    else if (it.getAsInteger(10, value))
      return Error(Twine("could not parse dimension integer from string: ") +
                   str + " because " + it + " cannot be parsed as an integer");
    shape.emplace_back(value);
  }

  return TensorShape(shape);
}
