//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoDWARFParser.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "MojoTypeSystem.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Utility/StreamString.h"
#include "llvm-project/lldb/source/Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "llvm-project/lldb/source/Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "llvm-project/lldb/source/Plugins/SymbolFile/DWARF/LogChannelDWARF.h"
#include <filesystem>

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;

// FIXME(23768): Remove this after the next LLVM integrate.
#define DIE_IS_BEING_PARSED ((lldb_private::Type *)1)

/// Bag of data for all the attributes parsed from a DWARF entry.
struct ParsedDWARFTypeAttributes {
  explicit ParsedDWARFTypeAttributes(const DWARFDIE &die) {
    DWARFAttributes attributes = die.GetAttributes();
    for (size_t i = 0; i < attributes.Size(); ++i) {
      dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue form_value;
      if (!attributes.ExtractFormValueAtIndex(i, form_value))
        continue;
      switch (attr) {
      default:
        break;

      case DW_AT_byte_size:
        byteSize = form_value.Unsigned();
        break;

      case DW_AT_alignment:
        alignment = form_value.Unsigned();
        break;

      case DW_AT_decl_file:
        decl.SetFile(
            attributes.CompileUnitAtIndex(i)->GetFile(form_value.Unsigned()));
        break;

      case DW_AT_decl_line:
        decl.SetLine(form_value.Unsigned());
        break;

      case DW_AT_decl_column:
        decl.SetColumn(form_value.Unsigned());
        break;

      case DW_AT_encoding:
        encoding = form_value.Unsigned();
        break;

      case DW_AT_external:
        external = form_value.Boolean();
        break;

      case DW_AT_inline:
        inlined = true;
        break;

      case DW_AT_linkage_name:
        mangledName.SetCString(form_value.AsCString());
        break;

      case DW_AT_name:
        name.SetCString(form_value.AsCString());
        break;

      case DW_AT_type:
        type = form_value;
        break;
      }
    }
  }

  bool inlined = false;
  lldb_private::ConstString mangledName;
  bool external = false;
  DWARFFormValue type;
  lldb_private::ConstString name;
  lldb_private::Declaration decl;
  std::optional<uint64_t> byteSize;
  std::optional<uint64_t> alignment;
  uint32_t encoding = 0;
};

/// Bag of data for all the attributes parsed from a struct member DWARF entry.
struct MemberAttributes {
  MemberAttributes(const DWARFDIE &die, const DWARFDIE &parentDie) {
    DWARFAttributes attributes = die.GetAttributes();
    for (size_t i = 0; i < attributes.Size(); ++i) {
      const dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue form_value;
      if (attributes.ExtractFormValueAtIndex(i, form_value)) {
        switch (attr) {
        case DW_AT_name:
          name = form_value.AsCString();
          break;
        case DW_AT_type:
          type = form_value;
          break;
        case DW_AT_data_member_location:
          byteOffset = form_value.Unsigned();
          break;
        default:
          break;
        }
      }
    }
  }
  const char *name = nullptr;
  std::optional<uint64_t> byteOffset;
  DWARFFormValue type;
};

MojoDWARFParser::MojoDWARFParser(MojoTypeSystem &typeSystem)
    : DWARFASTParser(Kind::DWARFASTParserClang), typeSystem(typeSystem) {}

MojoDWARFParser::~MojoDWARFParser() = default;

MojoASTDeclRef MojoDWARFParser::getDeclForDIE(const DWARFDIE &die) {
  if (!die)
    return {};
  if (MojoASTDeclRef decl = getCachedDeclForDIE(die))
    return decl;

  SymbolFileDWARF *dwarf = die.GetDWARF();
  MojoASTDeclRef decl;

  switch (die.Tag()) {
  case DW_TAG_compile_unit: {
    ParsedDWARFTypeAttributes attrs(die);
    std::string name;
    // The DIE name is the file path
    if (attrs.name.IsEmpty()) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::getDeclForDIE]: Compile unit name is empty. Die = "
          "{0:x16}.",
          die.GetOffset());
      name = "anonymous";
    } else {
      // The name of the module is the base name of the file. We might need to
      // eventually define a unique parser for each compile unit in case of name
      // collision.
      name = std::filesystem::path(attrs.name.AsCString()).stem().string();
    }

    decl = typeSystem.getOrCreateModuleDecl(name);
    break;
  }
  case DW_TAG_subprogram: {
    ParsedDWARFTypeAttributes attrs(die);
    decl = typeSystem.getOrCreateFunctionDecl(attrs.mangledName);
    break;
  }
  case DW_TAG_structure_type: {
    ParsedDWARFTypeAttributes attrs(die);
    std::string name;
    if (attrs.name.IsEmpty()) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::getDeclForDIE]: Structure name is empty. Die = "
          "{0:x16}.",
          die.GetOffset());
      name = "anonymous";
    } else {
      name = attrs.name;
    }

    decl = typeSystem.getOrCreateStructDecl(name, die);
    break;
  }
  default:
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::getDeclForDIE]: Unhandled type tag. Die = {0:x16}, "
        "tag = {1}.",
        die.GetOffset(), die.GetTagAsCString());
    break;
  }

  if (decl)
    dieToDecl[die.GetDIE()] = decl;

  return decl;
}

MojoASTDeclRef MojoDWARFParser::getCachedDeclForDIE(const DWARFDIE &die) {
  if (die) {
    if (auto pos = dieToDecl.find(die.GetDIE()); pos != dieToDecl.end())
      return pos->second;
  }
  return {};
}

MojoASTDeclRef
MojoDWARFParser::getDeclContextContainingDIE(const DWARFDIE &die,
                                             DWARFDIE *declDieCopy) {
  SymbolFileDWARF *dwarf = die.GetDWARF();

  DWARFDIE declDie = dwarf->GetDeclContextDIEContainingDIE(die);

  if (declDieCopy)
    *declDieCopy = declDie;

  if (declDie) {
    if (MojoASTDeclRef decl = getDeclForDIE(declDie))
      return decl;
  }
  return {};
}

void MojoDWARFParser::updateSymbolContextScopeForType(const SymbolContext &sc,
                                                      const DWARFDIE &die,
                                                      TypeSP &type) {
  assert(type->GetFullCompilerType().IsValid() &&
         "All types created from DWARF must be valid.");

  DWARFDIE scParentDie = SymbolFileDWARF::GetParentSymbolContextDIE(die);
  dw_tag_t scParentTag = scParentDie.Tag();

  SymbolContextScope *symbolContextScope = nullptr;
  if (scParentTag == DW_TAG_compile_unit) {
    symbolContextScope = sc.comp_unit;
  } else if (sc.function != nullptr && scParentDie) {
    symbolContextScope =
        sc.function->GetBlock(true).FindBlockByID(scParentDie.GetID());
    if (symbolContextScope == nullptr)
      symbolContextScope = sc.function;
  } else {
    symbolContextScope = sc.module_sp.get();
  }

  if (symbolContextScope != nullptr)
    type->SetSymbolContextScope(symbolContextScope);
}

CompilerDeclContext
MojoDWARFParser::GetDeclContextForUIDFromDWARF(const DWARFDIE &die) {
  if (die.Tag() == DW_TAG_compile_unit) {
    if (MojoASTDeclRef decl = getDeclForDIE(die))
      return CompilerDeclContext(&typeSystem, &decl);
  }
  return {};
}

TypeSP
MojoDWARFParser::ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                    const DWARFDIE &die, bool *typeIsNewPtr) {
  if (typeIsNewPtr)
    *typeIsNewPtr = false;

  if (!die)
    return {};

  SymbolFileDWARF *dwarf = die.GetDWARF();
  lldb_private::Type *typePtr = dwarf->GetDIEToType().lookup(die.GetDIE());
  if (typePtr == DIE_IS_BEING_PARSED)
    return nullptr;
  if (typePtr)
    return typePtr->shared_from_this();
  // Set a bit that lets us know that we are currently parsing this.
  dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;

  ParsedDWARFTypeAttributes attrs(die);

  if (Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups)) {
    dwarf->GetObjectFile()->GetModule()->LogMessage(
        log,
        "[MojoDWARFParser::ParseTypeFromDWARF] Will parse type. Die = {0:x16}, "
        "tag = {1}, name = '{2}', linkage name = '{3}', byte size = {4}.",
        die.GetOffset(), die.GetTagAsCString(), die.GetName(),
        attrs.mangledName.AsCString(), attrs.byteSize);
  }

  if (typeIsNewPtr)
    *typeIsNewPtr = true;

  const dw_tag_t tag = die.Tag();
  TypeSP type;

  switch (tag) {
  case DW_TAG_pointer_type: {
    DWARFDIE typeDie = attrs.type.Reference();
    lldb_private::Type *elementType =
        dwarf->ResolveTypeUID(typeDie, /*assert_not_being_parsed=*/true);
    CompilerType mojoType = typeSystem.GetPointerType(
        elementType->GetFullCompilerType().GetOpaqueQualType());

    type =
        dwarf->MakeType(die.GetID(), attrs.name, attrs.byteSize, nullptr,
                        attrs.type.Reference().GetID(),
                        lldb_private::Type::eEncodingIsPointerUID, &attrs.decl,
                        mojoType, lldb_private::Type::ResolveState::Full);
    break;
  }
  case DW_TAG_base_type: {
    if (attrs.byteSize.value_or(0) == 0) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::ParseTypeFromDWARF] Builtin type with 0 byte "
          "size. Die = {0:x16}, tag = {1}, name = '{2}'.",
          die.GetOffset(), die.GetTagAsCString(), die.GetName());
      break;
    }
    CompilerType mojoType = typeSystem.getBuiltinScalarType(
        attrs.name.GetStringRef(), attrs.encoding, attrs.byteSize.value_or(0));
    if (!mojoType.IsValid()) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "[MojoDWARFParser::ParseTypeFromDWARF] Couldn't create builtin type. "
          "Die = {0:x16}, tag = {1}, name = '{2}'.",
          die.GetOffset(), die.GetTagAsCString(), die.GetName());
      break;
    }
    type = dwarf->MakeType(die.GetID(), attrs.name, attrs.byteSize, nullptr,
                           attrs.type.Reference().GetID(),
                           lldb_private::Type::eEncodingIsUID, &attrs.decl,
                           mojoType, lldb_private::Type::ResolveState::Full);

    break;
  }
  case DW_TAG_subprogram: {
    if (MojoASTDeclRef decl = getDeclForDIE(die)) {
      CompilerType mojoType = typeSystem.createCompilerType(decl.getType());

      type = dwarf->MakeType(die.GetID(), attrs.name, attrs.byteSize, nullptr,
                             attrs.type.Reference().GetID(),
                             lldb_private::Type::eEncodingIsUID, &attrs.decl,
                             mojoType, lldb_private::Type::ResolveState::Full);
    }
    break;
  }
  case DW_TAG_array_type: {
    // The only array type we emit is simd, so we try to create one from the
    // given data. A pack could also work, but its type would be extremely
    // verbose.
    DWARFDIE dtypeDie = attrs.type.Reference();
    ParsedDWARFTypeAttributes dtypeAttrs(dtypeDie);

    std::optional<SymbolFile::ArrayInfo> arrayInfo = ParseChildArrayInfo(die);
    if (arrayInfo && !arrayInfo->element_orders.empty()) {
      size_t numElements = arrayInfo->element_orders.front();
      CompilerType elementType =
          typeSystem.createCompilerTypeFromDType(dtypeAttrs.name);
      if (elementType.IsValid()) {
        // LLDB expects us to provide a lldb_private::Type for the element type
        // of the SIMD.
        TypeSP lldbDType = dwarf->MakeType(
            dtypeDie.GetID(), dtypeAttrs.name, *dtypeAttrs.byteSize, nullptr,
            dtypeDie.GetID(), lldb_private::Type::eEncodingIsUID,
            &dtypeAttrs.decl, elementType,
            lldb_private::Type::ResolveState::Full);
        CompilerType mojoType =
            typeSystem.createSIMDType(dtypeDie.GetName(), numElements);
        if (mojoType.IsValid()) {
          type = dwarf->MakeType(
              die.GetID(), ConstString(),
              attrs.byteSize.value_or(*dtypeAttrs.byteSize * numElements),
              nullptr, die.GetID(), lldb_private::Type::eEncodingIsUID,
              &attrs.decl, mojoType, lldb_private::Type::ResolveState::Full);
          type->SetEncodingType(lldbDType.get());
        }
      }
    }
    if (!type) {
      dwarf->GetObjectFile()->GetModule()->ReportError(
          "The array type at offset {0} with element type '{1}' couldn't be "
          "parsed as a SIMD type.",
          die.GetOffset(), dtypeDie.GetName());
    }

    break;
  }
  case DW_TAG_structure_type: {
    // Several builtin types like !kgen.string are encoded as structs. We can
    // just parse them as regular MLIR types instead of traversing their DWARF.
    // At least in the specific case of primitive types like !kgen.string, it
    // will allow us to format them correctly because the corresponding printers
    // are type-based and not name-based.
    CompilerType mojoType =
        typeSystem.getBuiltinTypeFromMLIRTypeName(attrs.name);
    // If we recover the type, we need to make sure that the encoded byte size
    // matches the one from MLIR. If there's a mismatch, then either the debug
    // info is wrong or the MLIR type in this version of the parser is different
    // from the one that produced the debug info, in which case we discard the
    // MLIR type and do regular DWARF parsing.
    if (mojoType.IsValid()) {
      std::optional<uint64_t> mlirByteSize =
          mojoType.GetByteSize(/*exe_scope=*/nullptr);
      if (!mlirByteSize) {
        mojoType = {};
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "The parsed MLIR structure type '{0}' has not byte size. The "
            "MLIR type won't be used and regular MLIR-agnostic DWARF parsing "
            "will be performed.",
            attrs.name.AsCString());
      } else if (attrs.byteSize && *attrs.byteSize != *mlirByteSize) {
        mojoType = {};
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "The parsed MLIR structure type '{0}' has a different size ({1}) "
            "than the one in the debug info ({2}). The MLIR type won't be used "
            "and regular MLIR-agnostic DWARF parsing will be performed.",
            attrs.name.AsCString(), *mlirByteSize, *attrs.byteSize);
      } else {
        type = dwarf->MakeType(
            die.GetID(), attrs.name, attrs.byteSize, nullptr, LLDB_INVALID_UID,
            lldb_private::Type::eEncodingIsUID, &attrs.decl, mojoType,
            lldb_private::Type::ResolveState::Full);
      }
    }
    if (!mojoType.IsValid()) {
      if (MojoASTDeclRef decl = getDeclForDIE(die)) {
        CompilerType mojoType = typeSystem.createCompilerType(decl.getType());
        type = dwarf->MakeType(
            die.GetID(), attrs.name, attrs.byteSize, nullptr, LLDB_INVALID_UID,
            lldb_private::Type::eEncodingIsUID, &attrs.decl, mojoType,
            lldb_private::Type::ResolveState::Full);
        // FIXME(23821): We need to complete the struct right away here because
        // the generic dwarf parser uses the clang typesystem to complete types,
        // which obviously wouldn't work for us. We'll eventually fix this,
        // which will make the dwarf parser lazy.
        CompleteTypeFromDWARF(die, type.get(), mojoType);
      }
    }
    break;
  }
  default:
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::ParseTypeFromDWARF]: Unhandled type tag. "
        "Die = {0:x16}, tag = {1}.",
        die.GetOffset(), tag, die.GetTagAsCString());
    break;
  }

  if (type) {
    updateSymbolContextScopeForType(sc, die, type);
    dwarf->GetDIEToType()[die.GetDIE()] = type.get();
  }
  return type;
}

bool MojoDWARFParser::CompleteStructureTypeFromDWARF(
    const DWARFDIE &die, lldb_private::Type *type, CompilerType &compilerType) {
  MojoASTDeclRef structDecl = getDeclForDIE(die);
  assert(structDecl && "All structs should have a decl.");

  if (completedDecls.contains(&*structDecl))
    return true;

  SymbolFileDWARF *dwarf = die.GetDWARF();

  for (DWARFDIE memberDie : die.children()) {
    if (memberDie.Tag() == DW_TAG_member) {
      MemberAttributes attrs(memberDie, die);
      lldb_private::Type *memberType =
          die.ResolveTypeUID(attrs.type.Reference());
      if (!memberType) {
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "[MojoDWARFParser::CompleteTypeFromDWARF]: Couldn't complete "
            "the struct type '{0}' because one of its fields couldn't be "
            "parsed. Die = {1:x16}, memberDie = {2:x16}.",
            die.GetName(), die.GetOffset(), memberDie.GetOffset());
        return false;
      }

      std::optional<uint64_t> typeSize = memberType->GetByteSize(nullptr);
      if (!typeSize) {
        dwarf->GetObjectFile()->GetModule()->ReportError(
            "[MojoDWARFParser::CompleteTypeFromDWARF]: Couldn't complete "
            "the struct type '{0}' because one of its fields has no size. Die "
            "= {1:x16}, member die = {2:x16}.",
            die.GetName(), die.GetOffset(), memberDie.GetOffset());
        return false;
      }
      typeSystem.addFieldToStruct(
          structDecl, attrs.name,
          memberType->GetFullCompilerType().GetOpaqueQualType());
    }
  }
  ParsedDWARFTypeAttributes attrs(die);
  if (attrs.byteSize && attrs.byteSize != compilerType.GetByteSize(
                                              /*exe_scope=*/nullptr)) {
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::CompleteTypeFromDWARF]: The struct type '{0}' "
        "doesn't have the same size as reported in the DWARF after type "
        "completion. Die = {1:x16}.",
        die.GetName(), die.GetOffset());
    return false;
  }
  completedDecls.insert(&*structDecl);
  return true;
}

bool MojoDWARFParser::CompleteTypeFromDWARF(const DWARFDIE &die,
                                            lldb_private::Type *type,
                                            CompilerType &compilerType) {
  if (!die)
    return false;

  if (die.Tag() == DW_TAG_structure_type)
    return CompleteStructureTypeFromDWARF(die, type, compilerType);

  SymbolFileDWARF *dwarf = die.GetDWARF();
  dwarf->GetObjectFile()->GetModule()->ReportError(
      "[MojoDWARFParser::CompleteTypeFromDWARF]: Couldn't complete die. Die = "
      "{0:x16}, tag = {1}.",
      die.GetOffset(), die.GetTagAsCString());
  return false;
}

Function *
MojoDWARFParser::ParseFunctionFromDWARF(CompileUnit &comp_unit,
                                        const DWARFDIE &die,
                                        const AddressRange &funcRange) {
  Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups);
  SymbolFileDWARF *dwarf = die.GetDWARF();

  if (log) {
    dwarf->GetObjectFile()->GetModule()->LogMessage(
        log,
        "[MojoDWARFParser::ParseFunctionFromDWARF] Will parse function. Die = "
        "{0:x16}, tag = {1}, name = '{2}'.",
        die.GetOffset(), die.GetTagAsCString(), die.GetName());
  }

  auto doWork = [&]() -> Function * {
    assert(funcRange.GetBaseAddress().IsValid());

    DWARFRangeList funcRanges;
    const char *name = nullptr;
    const char *mangled = nullptr;
    std::optional<int> declFile;
    std::optional<int> declLine;
    std::optional<int> declColumn;
    std::optional<int> callFile;
    std::optional<int> callLine;
    std::optional<int> callColumn;
    DWARFExpressionList frameBase;

    const dw_tag_t tag = die.Tag();

    if (tag != DW_TAG_subprogram)
      return nullptr;

    if (die.GetDIENamesAndRanges(name, mangled, funcRanges, declFile, declLine,
                                 declColumn, callFile, callLine, callColumn,
                                 &frameBase)) {
      Mangled funcName;
      if (mangled)
        funcName.SetMangledName(ConstString(mangled));
      else
        funcName.SetMangledName(ConstString(name));
      funcName.SetDemangledName(ConstString(name));

      FunctionSP func;
      std::unique_ptr<Declaration> decl;
      if (declFile || declLine || declColumn)
        decl = std::make_unique<Declaration>(
            die.GetCU()->GetFile(declFile ? *declFile : 0),
            declLine ? *declLine : 0, declColumn ? *declColumn : 0);

      SymbolFileDWARF *dwarf = die.GetDWARF();
      // Supply the type only if it has already been parsed
      lldb_private::Type *funcType = dwarf->GetDIEToType().lookup(die.GetDIE());

      assert(funcType == nullptr || funcType != DIE_IS_BEING_PARSED);

      const user_id_t funcUID = die.GetID();
      func = std::make_shared<Function>(&comp_unit,
                                        funcUID, // UserID is the DIE offset
                                        funcUID, funcName, funcType,
                                        funcRange); // first address range

      if (func.get() != nullptr) {
        if (frameBase.IsValid())
          func->GetFrameBaseExpression() = frameBase;
        comp_unit.AddFunction(func);
        return func.get();
      }
    }
    return nullptr;
  };
  auto func = doWork();
  if (!func) {
    dwarf->GetObjectFile()->GetModule()->ReportError(
        "[MojoDWARFParser::ParseFunctionFromDWARF] failed to create a "
        "function. Die = {0:x16}, tag = {1}, name = '{2}'.",
        die.GetOffset(), die.GetTagAsCString(), die.GetName());
  }
  return func;
}
