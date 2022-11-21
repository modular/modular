//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGINFOENUMS_TD
#define DEBUGINFOENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// DIEmissionKind
//===----------------------------------------------------------------------===//

def DIEmissionNone                : I64EnumAttrCase<"None", 0>;
def DIEmissionFull                : I64EnumAttrCase<"Full", 1>;
def DIEmissionLineTablesOnly      : I64EnumAttrCase<"LineTablesOnly", 2>;
def DIEmissionDebugDirectivesOnly : I64EnumAttrCase<"DebugDirectivesOnly", 3>;

def DIEmissionKind : I64EnumAttr<
    "EmissionKind",
    "Debug emission kind", [
      DIEmissionNone,
      DIEmissionFull,
      DIEmissionLineTablesOnly,
      DIEmissionDebugDirectivesOnly,
    ]> {
  let cppNamespace = "::M::DebugInfo";
}

//===----------------------------------------------------------------------===//
// DISubprogramFlags
//===----------------------------------------------------------------------===//

def DISPVirtual        : I32BitEnumAttrCaseBit<"Virtual", 0>;
def DISPPureVirtual    : I32BitEnumAttrCaseBit<"PureVirtual", 1>;
def DISPLocalToUnit    : I32BitEnumAttrCaseBit<"LocalToUnit", 2>;
def DISPDefinition     : I32BitEnumAttrCaseBit<"Definition", 3>;
def DISPOptimized      : I32BitEnumAttrCaseBit<"Optimized", 4>;
def DISPPure           : I32BitEnumAttrCaseBit<"Pure", 5>;
def DISPElemental      : I32BitEnumAttrCaseBit<"Elemental", 6>;
def DISPRecursive      : I32BitEnumAttrCaseBit<"Recursive", 7>;
def DISPMainSubprogram : I32BitEnumAttrCaseBit<"MainSubprogram", 8>;
def DISPDeleted        : I32BitEnumAttrCaseBit<"Deleted", 9>;
def DISPObjCDirect     : I32BitEnumAttrCaseBit<"ObjCDirect", 11>;

def DISubprogramFlags : I32BitEnumAttr<
    "SubprogramFlags",
    "Subprogram specific flags", [
      DISPVirtual,
      DISPPureVirtual,
      DISPLocalToUnit,
      DISPDefinition,
      DISPOptimized,
      DISPPure, 
      DISPElemental,
      DISPRecursive,
      DISPMainSubprogram,
      DISPDeleted,
      DISPObjCDirect
    ]> {
  let cppNamespace = "::M::DebugInfo";
  let printBitEnumPrimaryGroups = 1;
}

#endif // DEBUGINFOENUMS_TD
