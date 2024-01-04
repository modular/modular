//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGINFOEXPRATTRS_TD
#define DEBUGINFOEXPRATTRS_TD

include "Support/DebugInfoDialect/IR/DebugInfoAttrs.td"
include "Support/DebugInfoDialect/IR/DebugInfoExprAttrInterfaces.td"

//===----------------------------------------------------------------------===//
// DIExprAttrBase
//===----------------------------------------------------------------------===//

class DIExprAttrBase<string attrName, string attrMnemonic,
  list<Trait> traits = []>
    : DIAttr<attrName, attrMnemonic,
             traits # [TypedAttrInterface, DebugInfo_DIExprAttrInterface]> {
  code diExprExtraClassDeclaration = [{}];
  let extraClassDeclaration = diExprExtraClassDeclaration # [{
    Type getType() const { return getDIType(); }
  }];
}

//===----------------------------------------------------------------------===//
// DIAggregatesIntoExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIAggregatesIntoExprAttr
  : DIExprAttrBase<"DIAggregatesIntoExpr", "expr.agg"> {
  let summary = "DI Expression fragment-to-aggregate operator.";
  let description = [{
    The `debuginfo.expr.agg` attribute represents a DI Expression operator
    that aggregates together a struct from a single field of that struct. Its
    operands are
      - A DI Expression of type T, and
      - A constant index i,
    and it has a DIStructType result where the i-th field has type T.

    Example:

    ```mlir
    !m1 = !debuginfo.member<foo: T>
    #ir_value = #debuginfo.expr.ir_value : T
    #agg = #debuginfo.expr.agg<#ir_value, 1> : !debuginfo.ti.struct<S(!m0, !m1)>
    ```
  }];
  let parameters = (ins
    "DIExprAttr":$fieldExpr,
    "unsigned":$fieldIndex,
    AttributeSelfTypeParameter<"", "::M::DebugInfo::DIStructType">:$DIType
  );
  let assemblyFormat = "`<` $fieldExpr `,` $fieldIndex `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "DIExprAttr":$fieldExpr,
                                        "unsigned":$fieldIndex,
                                        "DIStructType":$type), [{
      return get(type.getContext(), fieldExpr, fieldIndex, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DIDerefExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIDerefExprAttr : DIExprAttrBase<"DIDerefExpr", "expr.deref"> {
  let summary = "DI Expression dereference operator.";
  let description = [{
    The `#debuginfo.expr.deref` attribute represents a DI Expression operator
    that performs dereference on a pointer. Its operand is a DI Expression of
    type DITargetIndependentPointer<T> or DIPointer<T>, and has result type T.

    Example:

    ```mlir
    #ir_value = #debuginfo.expr.ir_value : !debuginfo.ti.ptr<T>
    #deref = #debuginfo.expr.deref<#ir_value>
    ```
  }];
  let parameters = (ins "DIExprAttr":$ptrExpr);
  let assemblyFormat = "`<` $ptrExpr `>`";
  let genVerifyDecl = 1;
  let diExprExtraClassDeclaration = [{ DIType getDIType() const; }];

  let builders = [TypeBuilderWithInferredContext<
    (ins "DIExprAttr":$ptrExpr), "", "DIExprAttr">];
}

//===----------------------------------------------------------------------===//
// DIIRValueExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIIRValueExprAttr
  : DIExprAttrBase<"DIIRValueExpr", "expr.irvalue"> {
  let summary = "DI Expression IR Value operator.";
  let description = [{
    The `#debuginfo.expr.ir_value` attribute represents a DI Expression operator
    that functions as a placeholder argument. During execution of the DI
    Expression, this value will be filled in based on the context. For example,
    when the surrounding DI Expression is used by a DebugInfo ValueOp, this
    attr represents the operand of that ValueOp.

    Example:

    ```mlir
    #location_expr = #debuginfo.expr.ir_value : index

    %0 = ... : index
    %1 = debuginfo.value #var #location_expr = %0 : index
    ```
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::M::DebugInfo::DIType">:$DIType
  );
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "DIType":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// DIRefOfExprAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIRefOfExprAttr : DIExprAttrBase<"DIRefOfExpr", "expr.refof"> {
  let summary = "DI Expression reference-of operator.";
  let description = [{
    The `#debuginfo.expr.refof` attribute represents a DI Expression operator
    that gets a reference from a value. Its operand is a DI Expression of any
    type T, and has result type DITargetIndependentPointer<T> or DIPointer<T>.

    Example:

    ```mlir
    #ir_value = #debuginfo.expr.ir_value : T
    #deref = #debuginfo.expr.refof<#ir_value> : !debuginfo.ti.ptr<T>
    ```
  }];
  let parameters = (ins
    "DIExprAttr":$valueExpr,
    AttributeSelfTypeParameter<"", "::M::DebugInfo::DIType">:$DIType
  );
  let assemblyFormat = "`<` $valueExpr `>`";

  let builders = [TypeBuilderWithInferredContext<
    (ins "DIExprAttr":$valueExpr, "DIType":$type), "", "DIExprAttr">];
}

#endif // DEBUGINFOEXPRATTRS_TD
