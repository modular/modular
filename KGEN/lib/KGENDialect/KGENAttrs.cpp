//===- KGENAttrs.cpp - Implement KGEN attributes --------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;

/// Parse a "colon type" production if present or default to si64 if not.  This
/// is commonly used in our parameter representation.
ParseResult KGEN::parseColonTypeOrSI64(OpAsmParser &parser, Type &type) {
  if (succeeded(parser.parseOptionalColon()))
    return parser.parseType(type);

  type = parser.getBuilder().getIntegerType(64, /*isSigned=*/true);
  return success();
}

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"

void KGENDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
      >();
}

Attribute KGENDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult = generatedAttributeParser(p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;

  p.emitError(p.getNameLoc(), "Unexpected kgen attribute '" + attrName + "'");
  return {};
}

void KGENDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(AsmParser &p, Type type) {
  if (type) {
    p.emitError(p.getNameLoc(), "unexpected contextual type in attribute");
    return {};
  }

  StringAttr name;
  if (p.parseLess() || p.parseAttribute(name, p.getBuilder().getNoneType()) ||
      p.parseColonType(type) || p.parseGreater())
    return {};

  return ParamDeclAttr::get(name, type);
}

void ParamDeclAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType() << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclRefAttr::parse(AsmParser &p, Type type) {
  std::string name;
  if (p.parseLess() || p.parseKeywordOrString(&name))
    return {};

  // If we have no contextual type then it must be present.
  if (!type) {
    if (p.parseColonType(type))
      return {};
  } else {
    // Otherwise it may be present, but must agree.
    Type explicitType;
    auto loc = p.getCurrentLocation();
    if (succeeded(p.parseOptionalColon())) {
      if (p.parseType(explicitType))
        return {};
      if (type != explicitType) {
        p.emitError(loc, "param decl ref '")
            << name << "' type " << explicitType
            << " didn't agree with contextual type" << type;
        return {};
      }
    }
  }

  if (p.parseGreater())
    return {};

  return ParamDeclRefAttr::get(name, type);
}

void ParamDeclRefAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType() << ">";
}

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

Attribute ParamBindAttr::parse(AsmParser &p, Type type) {
  std::string name;
  Attribute value;
  if (p.parseLess() || p.parseKeywordOrString(&name) ||
      p.parseColonType(type) || p.parseEqual() ||
      p.parseAttribute(value, type) || p.parseGreater())
    return {};

  return ParamBindAttr::get(name, type, value);
}

void ParamBindAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType() << " = ";
  p.printAttributeWithoutType(getValue());
  p << ">";
}

//===----------------------------------------------------------------------===//
// Parameter Verification
//===----------------------------------------------------------------------===//

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
static LogicalResult collectParameterUses(
    Attribute attr, Operation *op,
    SmallVectorImpl<std::pair<ParamDeclRefAttr, Operation *>> &parameterUses,
    llvm::SmallDenseSet<Attribute> &parameterLessAttrs) {

  // Reject errant parameter decls.
  if (auto paramDecl = attr.dyn_cast<ParamDeclAttr>()) {
    op->emitError("invalid ParamDeclAttr outside of paramDecls attribute ")
        << paramDecl;
    return failure();
  }

  // Collect parameter references.
  if (auto paramRef = attr.dyn_cast<ParamDeclRefAttr>()) {
    parameterUses.push_back({paramRef, op});
    return success();
  }

  // If this attribute has no sub-attributes or we have already scanned it an
  // know that it has no parameters in it, return early.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr, TypeAttr>() ||
      // TODO: Handle TypeAttr for parameterized types.
      parameterLessAttrs.count(attr))
    return success();

  // Otherwise we need to recursively process attributes that we know about.
  size_t oldSize = parameterUses.size();
  if (auto array = attr.dyn_cast<ArrayAttr>()) {
    for (auto elt : array)
      if (failed(
              collectParameterUses(elt, op, parameterUses, parameterLessAttrs)))
        return failure();
  } else if (auto bind = attr.dyn_cast<ParamBindAttr>()) {
    if (failed(collectParameterUses(bind.getValue(), op, parameterUses,
                                    parameterLessAttrs)))
      return failure();
  } else {
    // FIXME: hard coding specific attributes is really problematic, doesn't
    // MLIR have a generic way to walk sub-attributes?
    return op->emitError("unknown attribute for parameterization: ") << attr;
    return failure();
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == parameterUses.size())
    parameterLessAttrs.insert(attr);

  return success();
}

/// Scan the body of the specified operation checking invariants on
/// parameters, diagnosing errors and returning failure if so.  This is used
/// by verifiers for ops with bodies, like kgen.generator.
LogicalResult KGEN::checkParametersInOpBody(Operation *topLevelOp) {
  // Start by doing a pass over the operation and all the operations in its body
  // to find the definitions and uses of parameters.

  // Parameter definitions, if any are present, should all be in a single
  // `paramDecls` attribute on an operation.  We restrict where declarations
  // can be found to make them easier to identify and work with.  Keep track of
  // all the parameters we find by their name, this allows detecting
  // redefinitions with different types.
  SmallDenseMap<StringAttr, std::pair<Operation *, ParamDeclAttr>> paramDecls;

  // Parameter uses can occur in any attribute and even in in types.  We collect
  // all the uses we see by their operation.  Remember that attributes are
  // uniqued, so the same ParamDeclRefAttr can be used by multiple operations,
  // or even multiple times in the same operation.
  SmallVector<std::pair<ParamDeclRefAttr, Operation *>> parameterUses;

  // This is slow and expensive so we need to memoize the attributes and types
  // we've already checked.
  llvm::SmallDenseSet<Attribute> parameterLessAttrs;
  // TODO: parameterLessTypes.

  bool hadError = false;
  topLevelOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *bodyOp) {
    // Scan all the attributes and types to look for uses of parameters.  We let
    // the walker scan the region hierarchy.
    for (const NamedAttribute &namedAttr : bodyOp->getAttrs()) {
      // We handle paramDecls below specially.
      if (namedAttr.getName().strref() == "paramDecls")
        continue;
      // Scan the attribute tree looking or parameter uses and reject unexpected
      // parameter definitions.
      if (failed(collectParameterUses(namedAttr.getValue(), bodyOp,
                                      parameterUses, parameterLessAttrs))) {
        hadError = true;
        break;
      }

      // TODO: Look into types when we support parameterized types.
    }

    // Ok, check for parameter declarations as well.
    auto arrayAttr = bodyOp->getAttrOfType<ArrayAttr>("paramDecls");
    if (!arrayAttr)
      return;

    for (Attribute attr : arrayAttr) {
      // All the members of this array must be ParamDeclAttr's.
      auto param = attr.dyn_cast<ParamDeclAttr>();
      if (!param) {
        bodyOp->emitError("unknown attribute kind in paramDecls list ") << attr;
        hadError = true;
        return;
      }

      // We cannot have any redefinitions.
      auto &opAndDeclAttr = paramDecls[param.getName()];
      if (opAndDeclAttr.first) {
        auto diag = bodyOp->emitError("redeclaration of parameter ")
                    << param.getName();
        diag.attachNote(opAndDeclAttr.first->getLoc())
            << "previous declaration here";
        hadError = true;
        return;
      }
      opAndDeclAttr = {bodyOp, param};
    }
  });

  if (hadError)
    return failure();

  // Ok, now that we know the set of parameters we have to process, verify that
  // the uses match up and that we have a proper partial order relationship
  // between of definitions and uses.
  for (auto &[paramRefAttr, usingOp] : parameterUses) {
    // Check the use is referring to a parameter that was defined.
    auto decl = paramDecls[paramRefAttr.getName()];
    if (!decl.first) {
      usingOp->emitError("invalid use of parameter with no declaration ")
          << paramRefAttr.getName();
      return failure();
    }

    // Check that the types of the uses match the defs.
    if (decl.second.getType() != paramRefAttr.getType()) {
      auto diag = usingOp->emitError("invalid reference to parameter ")
                  << paramRefAttr;
      diag.attachNote(decl.first->getLoc())
          << "parameter defined as " << decl.second;
      return failure();
    }

    // FIXME: Check partial ordering.
  }

  return success();
}
