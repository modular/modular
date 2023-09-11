//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoDynamicVectorTypeFormatter.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

MojoDynamicVectorSyntheticFrontEnd::MojoDynamicVectorSyntheticFrontEnd(
    lldb::ValueObjectSP backend)
    : SyntheticChildrenFrontEnd(*backend), start(0), size(0), elementType(),
      elementSize(0) {
  if (backend)
    Update();
}

MojoDynamicVectorSyntheticFrontEnd::~MojoDynamicVectorSyntheticFrontEnd() {}

size_t MojoDynamicVectorSyntheticFrontEnd::CalculateNumChildren() {
  return size;
}

lldb::ValueObjectSP
MojoDynamicVectorSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (idx >= size)
    return ValueObjectSP();
  uint64_t addr = start + (idx * elementSize);
  return CreateValueObjectFromAddress(llvm::formatv("[{0}]", idx).str(), addr,
                                      m_backend.GetExecutionContextRef(),
                                      elementType);
}

bool MojoDynamicVectorSyntheticFrontEnd::Update() {
  // Extract data pointer.
  lldb::ValueObjectSP dataField = m_backend.GetChildMemberWithName("data");
  if (!dataField)
    return false;

  lldb::ValueObjectSP kgenPointer =
      dataField->GetChildMemberWithName("address");

  if (!kgenPointer || !kgenPointer->IsPointerType())
    return false;

  start = kgenPointer->GetPointerValue();
  if (start == LLDB_INVALID_ADDRESS)
    return false;

  // Get element type and size.
  elementType = kgenPointer->GetCompilerType().GetPointeeType();
  auto exeCtxScope = ExecutionContext(m_backend.GetExecutionContextRef())
                         .GetBestExecutionContextScope();
  if (auto eltSize = elementType.GetByteSize(exeCtxScope))
    elementSize = eltSize.value();
  else
    return false;

  // Get the size.value field.
  lldb::ValueObjectSP sizeField = m_backend.GetChildMemberWithName("size");
  if (!sizeField)
    return false;

  lldb::ValueObjectSP sizeVal = sizeField->GetChildMemberWithName("value");
  if (!sizeVal)
    return false;

  bool success = false;
  uint64_t maybeSize = sizeVal->GetValueAsUnsigned(0, &success);

  if (success)
    size = maybeSize;
  return false;
}

bool MojoDynamicVectorSyntheticFrontEnd::MightHaveChildren() { return true; }

size_t MojoDynamicVectorSyntheticFrontEnd::GetIndexOfChildWithName(
    lldb_private::ConstString name) {
  if (size == 0)
    return 0;
  return ExtractIndexFromString(name.GetCString());
}

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::MojoDynamicVectorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  CompilerType type = valobjSP->GetCompilerType();
  if (!type.IsValid())
    return nullptr;
  return new M::KGEN::Mojo::MojoDynamicVectorSyntheticFrontEnd(valobjSP);
}
