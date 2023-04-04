//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONVARIABLE_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONVARIABLE_H

#include "lldb/Core/Value.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-public.h"
#include "llvm/Support/Casting.h"

namespace M::KGEN::Mojo {
//===----------------------------------------------------------------------===//
// MojoExpressionVariable
//===----------------------------------------------------------------------===//

/// This class represents a single Mojo expression variable.
class MojoExpressionVariable : public lldb_private::ExpressionVariable {
public:
  MojoExpressionVariable(lldb_private::ExecutionContextScope *exeScope,
                         lldb::ByteOrder byteOrder, uint32_t addrByteSize);
  MojoExpressionVariable(const lldb::ValueObjectSP &valobj);
  MojoExpressionVariable(lldb_private::ExecutionContextScope *exeScope,
                         lldb_private::ConstString name,
                         const lldb_private::TypeFromUser &userType,
                         lldb::ByteOrder byteOrder, uint32_t addrByteSize);

  // Prevent copying.
  MojoExpressionVariable(const MojoExpressionVariable &) = delete;
  const MojoExpressionVariable &
  operator=(const MojoExpressionVariable &) = delete;

  //--------------------------------------------------------------------------//
  // llvm casting support
  //--------------------------------------------------------------------------//

  static LLVMCastKind classofKind() {
    // TODO: MojoExpressionVariable should use a more open casting mechanism,
    // but for now just pretend we are Go.
    return LLVMCastKind::eKindGo;
  }

  static bool classof(const ExpressionVariable *ev) {
    return ev->getKind() == classofKind();
  }
};

//===----------------------------------------------------------------------===//
// MojoPersistentExpressionState
//===----------------------------------------------------------------------===//

/// This class manages persistent values that need to be preserved between
/// Mojo expression invocations.
class MojoPersistentExpressionState
    : public lldb_private::PersistentExpressionState {

public:
  MojoPersistentExpressionState() : PersistentExpressionState(classofKind()) {}
  ~MojoPersistentExpressionState() override = default;

  lldb::ExpressionVariableSP
  CreatePersistentVariable(const lldb::ValueObjectSP &valobj) override;

  lldb::ExpressionVariableSP
  CreatePersistentVariable(lldb_private::ExecutionContextScope *exeScope,
                           lldb_private::ConstString name,
                           const lldb_private::CompilerType &compilerType,
                           lldb::ByteOrder byteOrder,
                           uint32_t addrByteSize) override;

  llvm::StringRef GetPersistentVariablePrefix(bool isError) const override {
    // TODO: This is a placeholder, and should be replaced when we actually
    // support persistent variables.
    return isError ? "$E" : "$R";
  }

  void RemovePersistentVariable(lldb::ExpressionVariableSP variable) override {}

  lldb_private::ConstString
  GetNextPersistentVariableName(bool isError = false) override {
    return lldb_private::ConstString("");
  }

  std::optional<lldb_private::CompilerType> GetCompilerTypeFromPersistentDecl(
      lldb_private::ConstString typeName) override {
    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // RTTI support
  //===--------------------------------------------------------------------===//

  static LLVMCastKind classofKind() {
    // TODO: PersistentExpressionState should use a more open casting mechanism,
    // but for now just pretend we are Go.
    return LLVMCastKind::eKindGo;
  }

  static bool classof(const PersistentExpressionState *pv) {
    return pv->getKind() == classofKind();
  }
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONVARIABLE_H
