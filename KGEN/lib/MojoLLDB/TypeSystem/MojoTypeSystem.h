//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H
#define KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H

#include "../Plugin.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-private.h"

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN::Mojo {
class MojoTypeSystem : public lldb_private::TypeSystem {
  static char ID;

public:
  MojoTypeSystem(lldb_private::Target &target);
  ~MojoTypeSystem() override;

  /// Return the MLIR context for this type system.
  MLIRContext *getMLIRContext();

  /// Return the LLCL runtime for this type system.
  LLCL::Runtime &getRuntime();

  /// Return if the given language is supported by this type system.
  bool SupportsLanguage(lldb::LanguageType language) override {
    return language == eLanguageTypeMojo;
  }

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  static void Initialize();
  static void Terminate();

  llvm::StringRef GetPluginName() override { return getPluginNameStatic(); }
  static llvm::StringRef getPluginNameStatic() { return "Mojo"; }

  //===--------------------------------------------------------------------===//
  // Dumping
  //===--------------------------------------------------------------------===//

  void Dump(llvm::raw_ostream &output) override {}

  void DumpValue(lldb::opaque_compiler_type_t type,
                 lldb_private::ExecutionContext *exeCtx,
                 lldb_private::Stream *s, lldb::Format format,
                 const lldb_private::DataExtractor &data,
                 lldb::offset_t dataOffset, size_t dataByteSize,
                 uint32_t bitfieldBitSize, uint32_t bitfieldBitOffset,
                 bool showTypes, bool showSummary, bool verbose,
                 uint32_t depth) override {}

  void DumpSummary(lldb::opaque_compiler_type_t type,
                   lldb_private::ExecutionContext *exeCtx,
                   lldb_private::Stream *s,
                   const lldb_private::DataExtractor &data,
                   lldb::offset_t dataOffset, size_t dataByteSize) override {}

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, lldb_private::Stream *s,
                     lldb::Format format,
                     const lldb_private::DataExtractor &data,
                     lldb::offset_t dataOffset, size_t dataByteSize,
                     uint32_t bitfieldBitSize, uint32_t bitfieldBitOffset,
                     lldb_private::ExecutionContextScope *exeScope) override {
    return {};
  }

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump(lldb::opaque_compiler_type_t type) const override {
  }
#endif

  /// Dump the type to stdout.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override {}

  /// Print a description of the type to a stream. The exact implementation
  /// varies, but the expectation is that eDescriptionLevelFull returns a
  /// source-like representation of the type, whereas eDescriptionLevelVerbose
  /// does a dump of the underlying AST if applicable.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, lldb_private::Stream *s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override {}

  //===--------------------------------------------------------------------===//
  // Type Queries
  //===--------------------------------------------------------------------===//

#ifndef NDEBUG
  /// Verify the integrity of the type to catch CompilerTypes that mix
  /// and match invalid TypeSystem/Opaque type pairs.
  bool Verify(lldb::opaque_compiler_type_t type) override {
    // MLIR type construction should already handle verifying the necessary
    // invariants here.
    return true;
  }
#endif

  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override {
    return eLanguageTypeMojo;
  }

  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override;

  bool GetCompleteType(lldb::opaque_compiler_type_t type) override {
    return true;
  }

  bool CanPassInRegisters(const lldb_private::CompilerType &type) override {
    return false;
  }

  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  const llvm::fltSemantics &GetFloatTypeSemantics(size_t byteSize) override {
    return llvm::APFloatBase::Bogus();
  }

  bool
  ShouldTreatScalarValueAsAddress(lldb::opaque_compiler_type_t type) override;

  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  lldb_private::CompilerType
  GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                             const size_t index) override {
    return {};
  }

  uint32_t GetPointerByteSize() override { return 8; }

  lldb_private::ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                                        bool baseOnly) override {
    return {};
  }

  lldb_private::ConstString
  GetDisplayTypeName(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  uint32_t GetTypeInfo(
      lldb::opaque_compiler_type_t type,
      lldb_private::CompilerType *pointeeOrElementCompilerType) override {
    return {};
  }
  /// An overload of GetTypeInfo that uses a null
  /// `pointeeOrElementCompilerType`.
  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type) {
    return GetTypeInfo(type, /*pointeeOrElementCompilerType=*/nullptr);
  }

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  std::optional<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             lldb_private::ExecutionContextScope *exeScope) override;

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override {
    return {};
  }

  //===--------------------------------------------------------------------===//
  // DeclContext
  //===--------------------------------------------------------------------===//

  lldb_private::ConstString DeclGetName(void *opaqueDecl) override {
    return {};
  }
  lldb_private::CompilerType GetTypeForDecl(void *opaqueDecl) override {
    return {};
  }
  lldb_private::ConstString DeclContextGetName(void *opaqueDeclCtx) override {
    return {};
  }
  lldb_private::ConstString
  DeclContextGetScopeQualifiedName(void *opaqueDeclCtx) override {
    return {};
  }
  bool DeclContextIsClassMethod(void *opaqueDeclCtx) override { return {}; }

  bool DeclContextIsContainedInLookup(void *opaqueDeclCtx,
                                      void *otherOpaqueDeclCtx) override {
    return {};
  }

  lldb::LanguageType DeclContextGetLanguage(void *) override {
    return eLanguageTypeMojo;
  }

  //===--------------------------------------------------------------------===//
  // IsType Queries
  //===--------------------------------------------------------------------===//

  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsCharType(lldb::opaque_compiler_type_t type) override { return false; }
  bool IsCompleteType(lldb::opaque_compiler_type_t type) override {
    return true;
  }
  bool IsConst(lldb::opaque_compiler_type_t type) override { return false; }
  bool IsFloatingPointType(lldb::opaque_compiler_type_t type, uint32_t &count,
                           bool &isComplex) override;
  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &isSigned) override;
  bool IsScopedEnumerationType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsScalarType(lldb::opaque_compiler_type_t type) override;
  bool IsCStringType(lldb::opaque_compiler_type_t type,
                     uint32_t &length) override {
    return false;
  }
  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    lldb_private::CompilerType *elementType,
                    uint64_t *size) override {
    return false;
  }
  uint32_t
  IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                         lldb_private::CompilerType *baseTypePtr) override {
    return 0;
  }
  bool IsBlockPointerType(
      lldb::opaque_compiler_type_t type,
      lldb_private::CompilerType *functionPointerTypePtr) override {
    return false;
  }
  bool IsMemberFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool
  IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                           lldb_private::CompilerType *pointeeType) override {
    return false;
  }

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       lldb_private::CompilerType *pointeeType,
                       bool *isRValue) override {
    return false;
  }

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   lldb_private::CompilerType *elementType, uint64_t *size,
                   bool *isIncomplete) override {
    return false;
  }

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsDefined(lldb::opaque_compiler_type_t type) override { return false; }

  bool IsFunctionType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             lldb_private::CompilerType *targetType,
                             bool checkCplusplus, bool checkObjc) override {
    return false;
  }

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     lldb_private::CompilerType *pointeeType) override {
    return false;
  }

  bool IsVoidType(lldb::opaque_compiler_type_t type) override { return false; }

  //===--------------------------------------------------------------------===//
  // GetType Queries
  //===--------------------------------------------------------------------===//

  lldb_private::CompilerType
  GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) override {
    return {};
  }
  lldb_private::CompilerType
  GetBasicTypeFromAST(lldb::BasicType basic_type) override {
    return {};
  }
  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override {
    return lldb::eBasicTypeInvalid;
  }

  lldb_private::CompilerType
  GetLValueReferenceType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetRValueReferenceType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetNonReferenceType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  lldb_private::ExecutionContextScope *exeScope) override {
    return {};
  }

  lldb_private::CompilerType
  GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                      size_t bitSize) override {
    return {};
  }

  lldb_private::CompilerType
  GetTypedefedType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetArrayElementType(lldb::opaque_compiler_type_t type,
                      lldb_private::ExecutionContextScope *exeScope) override {
    return {};
  }

  lldb_private::CompilerType GetArrayType(lldb::opaque_compiler_type_t type,
                                          uint64_t size) override {
    return {};
  }

  lldb_private::CompilerType
  GetCanonicalType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override {
    return -1;
  }

  lldb_private::CompilerType
  GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                 size_t idx) override {
    return {};
  }

  lldb_private::CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override {
    return {};
  }

  lldb_private::CompilerType
  GetPointeeType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetPointerType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  //===--------------------------------------------------------------------===//
  // Type Navigation
  //===--------------------------------------------------------------------===//

  uint32_t
  GetNumChildren(lldb::opaque_compiler_type_t type, bool omitEmptyBaseClasses,
                 const lldb_private::ExecutionContext *exeCtx) override {
    return 0;
  }

  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  lldb_private::CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type,
                                             size_t idx, std::string &name,
                                             uint64_t *bitOffsetPtr,
                                             uint32_t *bitfieldBitSizePtr,
                                             bool *isBitfieldPtr) override {
    return {};
  }

  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  lldb_private::CompilerType
  GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                            uint32_t *bitOffsetPtr) override {
    return {};
  }

  lldb_private::CompilerType
  GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                             uint32_t *bitOffsetPtr) override {
    return {};
  }

  lldb_private::CompilerType GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, lldb_private::ExecutionContext *exeCtx,
      size_t idx, bool transparent_pointers, bool omitEmptyBaseClasses,
      bool ignoreArrayBounds, std::string &childName, uint32_t &childByteSize,
      int32_t &childByteOffset, uint32_t &childBitfieldBitSize,
      uint32_t &childBitfieldBitOffset, bool &childIsBaseClass,
      bool &childIsDerefOfParent, lldb_private::ValueObject *valobj,
      uint64_t &languageFlags) override {
    return {};
  }

  uint32_t GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                   const char *name,
                                   bool omitEmptyBaseClasses) override {
    return 0;
  }

  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                const char *name, bool omitEmptyBaseClasses,
                                std::vector<uint32_t> &childIndexes) override {
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  /// Return a new user expression for the given expression text, or nullptr in
  /// the case of an error.
  lldb_private::UserExpression *
  GetUserExpression(StringRef expr, StringRef prefix,
                    lldb::LanguageType language,
                    lldb_private::Expression::ResultType desiredType,
                    const lldb_private::EvaluateExpressionOptions &options,
                    lldb_private::ValueObject *ctxObj) override;

  /// Return a pointer to the persistent expression state for this type system.
  lldb_private::PersistentExpressionState *
  GetPersistentExpressionState() override;

  //===--------------------------------------------------------------------===//
  // RTTI support
  //===--------------------------------------------------------------------===//

  bool isA(const void *classID) const override { return classID == &ID; }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }

protected:
  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H
