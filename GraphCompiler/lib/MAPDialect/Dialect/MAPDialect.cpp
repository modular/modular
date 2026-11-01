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

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MAP;

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

void MAPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SliceSpecAttr
//===----------------------------------------------------------------------===//

// Parses `[` (`?` | int) (`,` (`?` | int))* `]`, with `?` for
// `ShapedType::kDynamic`.
static ParseResult parseSliceSpecBounds(AsmParser &parser,
                                        SmallVectorImpl<int64_t> &out) {
  if (parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  do {
    if (succeeded(parser.parseOptionalQuestion())) {
      out.push_back(ShapedType::kDynamic);
      continue;
    }
    int64_t value;
    if (parser.parseInteger(value))
      return failure();
    out.push_back(value);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRSquare();
}

static void printSliceSpecBounds(AsmPrinter &printer,
                                 ArrayRef<int64_t> values) {
  printer << '[';
  llvm::interleaveComma(values, printer, [&](int64_t v) {
    if (ShapedType::isDynamic(v))
      printer << '?';
    else
      printer << v;
  });
  printer << ']';
}

static ParseResult parseSliceSpecBoundsField(AsmParser &parser,
                                             StringRef keyword,
                                             SmallVectorImpl<int64_t> &out) {
  if (parser.parseKeyword(keyword) || parser.parseEqual())
    return failure();
  return parseSliceSpecBounds(parser, out);
}

//===----------------------------------------------------------------------===//
// ViewSignatureAttr
//===----------------------------------------------------------------------===//

// Parses `[in-shape] -> [out-shape]` without the surrounding `<...>`, so the
// spec attrs can embed it inline.
static ParseResult parseViewSignatureBody(AsmParser &parser,
                                          MOSH::ShapeAttr &in,
                                          MOSH::ShapeAttr &out) {
  auto parseShape = [&](MOSH::ShapeAttr &shape) -> ParseResult {
    TypedAttr shapeAttr;
    auto result = MOSH::parseShapeAttr(parser, shapeAttr);
    if (!result.has_value())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '[' for shape");
    if (failed(*result))
      return failure();
    shape = llvm::cast<MOSH::ShapeAttr>(shapeAttr);
    return success();
  };
  return failure(failed(parseShape(in)) || parser.parseArrow() ||
                 failed(parseShape(out)));
}

static void printViewSignatureBody(AsmPrinter &printer, MOSH::ShapeAttr in,
                                   MOSH::ShapeAttr out) {
  MOSH::printShapeTypedAttr(printer, in);
  printer << " -> ";
  MOSH::printShapeTypedAttr(printer, out);
}

Attribute ViewSignatureAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr in, out;
  if (parser.parseLess() || parseViewSignatureBody(parser, in, out) ||
      parser.parseGreater())
    return {};
  return ViewSignatureAttr::get(parser.getContext(), in, out);
}

void ViewSignatureAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getIn(), getOut());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// SliceSpecAttr
//===----------------------------------------------------------------------===//

llvm::SmallVector<int64_t> MAP::inferSliceSteps(MOSH::ShapeAttr out,
                                                llvm::ArrayRef<int64_t> starts,
                                                llvm::ArrayRef<int64_t> stops) {
  SmallVector<int64_t> outDims = out.getIntDims();
  llvm::SmallVector<int64_t> steps;
  steps.reserve(outDims.size());
  for (size_t i = 0, e = outDims.size(); i < e; ++i) {
    if (ShapedType::isDynamic(outDims[i]) || ShapedType::isDynamic(starts[i]) ||
        ShapedType::isDynamic(stops[i])) {
      steps.push_back(ShapedType::kDynamic);
      continue;
    }
    int64_t span = stops[i] - starts[i];
    if (outDims[i] == 0 || span % outDims[i] != 0) {
      steps.push_back(ShapedType::kDynamic);
      continue;
    }
    steps.push_back(span / outDims[i]);
  }
  return steps;
}

Attribute SliceSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  SmallVector<int64_t> starts, stops, steps;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseComma() ||
      parseSliceSpecBoundsField(parser, "starts", starts) ||
      parser.parseComma() ||
      parseSliceSpecBoundsField(parser, "stops", stops) ||
      parser.parseComma() ||
      parseSliceSpecBoundsField(parser, "steps", steps) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<SliceSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape),
      mlir::DenseI64ArrayAttr::get(ctx, starts),
      mlir::DenseI64ArrayAttr::get(ctx, stops),
      mlir::DenseI64ArrayAttr::get(ctx, steps));
}

void SliceSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ", starts = ";
  printSliceSpecBounds(printer, getStarts().asArrayRef());
  printer << ", stops = ";
  printSliceSpecBounds(printer, getStops().asArrayRef());
  printer << ", steps = ";
  printSliceSpecBounds(printer, getSteps().asArrayRef());
  printer << ">";
}

LogicalResult SliceSpecAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    MAP::ViewSignatureAttr signature, mlir::DenseI64ArrayAttr starts,
    mlir::DenseI64ArrayAttr stops, mlir::DenseI64ArrayAttr steps) {
  MOSH::ShapeAttr in = signature.getIn();
  MOSH::ShapeAttr out = signature.getOut();
  // ViewSignatureAttr allows rank changes; slice does not.
  if (in.getRank() != out.getRank())
    return emitError() << "in/out shapes must have the same rank, got "
                       << in.getRank() << " and " << out.getRank();

  int64_t rank = out.getRank();
  if (starts.size() != rank || stops.size() != rank || steps.size() != rank)
    return emitError() << "starts/stops/steps must each have " << rank
                       << " entries (one per dimension), got " << starts.size()
                       << "/" << stops.size() << "/" << steps.size();

  SmallVector<int64_t> inDims = in.getIntDims();
  SmallVector<int64_t> outDims = out.getIntDims();
  ArrayRef<int64_t> startVals = starts.asArrayRef();
  ArrayRef<int64_t> stopVals = stops.asArrayRef();
  ArrayRef<int64_t> stepVals = steps.asArrayRef();
  for (int64_t i = 0; i < rank; ++i) {
    // A dimension with any unknown bound is left to MO's own shape function,
    // which raises at run time -- see E2E/raising-shape-funcs.mlir, which
    // compiles a zero step on purpose and expects the runtime diagnostic.
    if (ShapedType::isDynamic(inDims[i]) ||
        ShapedType::isDynamic(startVals[i]) ||
        ShapedType::isDynamic(stopVals[i]))
      continue;
    bool stepKnown = !ShapedType::isDynamic(stepVals[i]);
    // A reverse slice walks down to -1, so its bounds sit one below the
    // forward range; an unknown step admits either.
    bool reverse = stepKnown && stepVals[i] < 0;
    int64_t lo = (!stepKnown || reverse) ? -1 : 0;
    int64_t hi = (!stepKnown || !reverse) ? inDims[i] : inDims[i] - 1;
    bool ordered = !stepKnown || (reverse ? startVals[i] >= stopVals[i]
                                          : startVals[i] <= stopVals[i]);
    if (startVals[i] < lo || startVals[i] > hi || stopVals[i] < lo ||
        stopVals[i] > hi || !ordered)
      return emitError() << "dim " << i << ": start/stop out of bounds for "
                         << "in-shape dim " << inDims[i]
                         << ", got start=" << startVals[i]
                         << ", stop=" << stopVals[i];

    if (ShapedType::isDynamic(outDims[i]) || !stepKnown)
      continue;
    if (stepVals[i] == 0)
      return emitError() << "dim " << i << ": step must be non-zero";
    if (reverse) {
      // Normalization maps a literal -1 stop to inDim-1, so a reverse stop
      // cannot express "past element 0" and the out shape is authoritative
      // instead. Check only that the walk stays in bounds.
      int64_t last = startVals[i] + (outDims[i] - 1) * stepVals[i];
      if (outDims[i] > 0 && (last < 0 || last >= inDims[i]))
        return emitError() << "dim " << i << ": start/step walk out of bounds "
                           << "for in-shape dim " << inDims[i]
                           << ", got start=" << startVals[i]
                           << ", step=" << stepVals[i] << ", out-shape dim "
                           << outDims[i];
      continue;
    }
    // MO sizes a slice by ceiling division (see computeRangeSize), so
    // start=0, stop=5, step=2 yields 3, not an exact multiple of the step.
    if (outDims[i] !=
        llvm::divideCeilSigned(stopVals[i] - startVals[i], stepVals[i]))
      return emitError() << "dim " << i << ": start/stop/step do not produce "
                         << "out-shape dim " << outDims[i]
                         << ", got start=" << startVals[i]
                         << ", stop=" << stopVals[i]
                         << ", step=" << stepVals[i];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastSpecAttr
//===----------------------------------------------------------------------===//

Attribute BroadcastSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<BroadcastSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape));
}

void BroadcastSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ">";
}

LogicalResult
BroadcastSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          MAP::ViewSignatureAttr signature) {
  MOSH::ShapeAttr in = signature.getIn();
  MOSH::ShapeAttr out = signature.getOut();
  // Broadcast can add rank (numpy semantics) -- never remove it.
  if (in.getRank() > out.getRank())
    return emitError() << "broadcast cannot reduce rank, got in-rank "
                       << in.getRank() << " > out-rank " << out.getRank();

  SmallVector<int64_t> inDims = in.getIntDims();
  SmallVector<int64_t> outDims = out.getIntDims();
  int64_t inRank = in.getRank();
  int64_t delta = static_cast<int64_t>(out.getRank()) - inRank;
  for (int64_t i = 0; i < inRank; ++i) {
    int64_t inDim = inDims[i];
    int64_t outDim = outDims[i + delta];
    if (ShapedType::isDynamic(inDim) || ShapedType::isDynamic(outDim))
      continue;
    if (inDim != 1 && inDim != outDim)
      return emitError() << "dim " << i << ": in-shape dim " << inDim
                         << " is not broadcastable to out-shape dim " << outDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeSpecAttr
//===----------------------------------------------------------------------===//

Attribute ReshapeSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<ReshapeSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape));
}

void ReshapeSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ">";
}

LogicalResult
ReshapeSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        MAP::ViewSignatureAttr signature) {
  // Only the element count must match, as in MO::Static::ReshapeOp.
  return MO::checkReshapeShapesCompatible(signature.getIn(), signature.getOut(),
                                          emitError);
}

//===----------------------------------------------------------------------===//
// TransposeSpecAttr
//===----------------------------------------------------------------------===//

Attribute TransposeSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  SmallVector<int64_t> perm;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseComma() || parseSliceSpecBoundsField(parser, "perm", perm) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<TransposeSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape),
      mlir::DenseI64ArrayAttr::get(ctx, perm));
}

void TransposeSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ", perm = ";
  printSliceSpecBounds(printer, getPerm().asArrayRef());
  printer << ">";
}

LogicalResult
TransposeSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          MAP::ViewSignatureAttr signature,
                          mlir::DenseI64ArrayAttr perm) {
  MOSH::ShapeAttr in = signature.getIn();
  MOSH::ShapeAttr out = signature.getOut();
  if (in.getRank() != out.getRank())
    return emitError() << "in/out shapes must have the same rank, got "
                       << in.getRank() << " and " << out.getRank();

  int64_t rank = out.getRank();
  if (static_cast<int64_t>(perm.size()) != rank)
    return emitError() << "perm must have " << rank
                       << " entries (one per dimension), got " << perm.size();

  SmallVector<int64_t> permVec(perm.asArrayRef());
  // A dynamic perm (the eager interpreter's one-graph-per-op transpose) is
  // resolved at runtime. In practice a perm is fully static or fully dynamic.
  bool fullyStatic = llvm::none_of(permVec, ShapedType::isDynamic);
  if (fullyStatic) {
    SmallVector<int64_t> sortedPerm(permVec);
    llvm::sort(sortedPerm);
    for (int64_t i = 0; i < rank; ++i) {
      if (sortedPerm[i] != i)
        return emitError() << "perm must be a permutation of [0, " << rank
                           << ")";
    }
  }

  SmallVector<int64_t> inDims = in.getIntDims();
  SmallVector<int64_t> outDims = out.getIntDims();
  for (int64_t i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(permVec[i]))
      continue;
    int64_t inDim = inDims[permVec[i]];
    int64_t outDim = outDims[i];
    if (ShapedType::isDynamic(inDim) || ShapedType::isDynamic(outDim))
      continue;
    if (inDim != outDim)
      return emitError() << "dim " << i << ": out-shape dim " << outDim
                         << " does not match in-shape dim " << inDim
                         << " at perm[" << i << "]=" << permVec[i];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
