//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Transforms/PackageUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Buffer.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// loadPreElaboratedModuleBytecode
//===----------------------------------------------------------------------===//

ErrorOr<OwningOpRef<ModuleOp>>
KGEN::loadPreElaboratedModuleBytecode(DenseResourceElementsAttr bytecodeAttr) {
  mlir::AsmResourceBlob *blob = bytecodeAttr.getRawHandle().getBlob();
  if (!blob)
    return Error("unable to find the pre-elaborated module blob");
  ArrayRef<char> bytecode = blob->getData();
  llvm::MemoryBufferRef bufferRef(StringRef(bytecode.begin(), bytecode.size()),
                                  "");

  // Read the entirety of the bytecode in the buffer; no lazy loading.
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  mlir::ParserConfig parserConfig(bytecodeAttr.getContext());
  mlir::BytecodeReader reader(bufferRef, parserConfig, /*lazyLoad=*/false,
                              sourceMgr);
  Block block;
  if (failed(reader.readTopLevel(&block)))
    return Error("unable to read pre-elaborated module blob");

  // Take ownership of the parsed module by removing it from the block so that
  // we can return it.
  ModuleOp module = cast<ModuleOp>(block.front());
  module->remove();
  return module;
}

//===----------------------------------------------------------------------===//
// loadPreElaboratedModuleForLinking
//===----------------------------------------------------------------------===//

ErrorOr<OwningOpRef<ModuleOp>> KGEN::loadPreElaboratedModuleForLinking(
    DenseResourceElementsAttr bytecodeAttr) {
  ErrorOr<OwningOpRef<ModuleOp>> packageModuleOr =
      loadPreElaboratedModuleBytecode(bytecodeAttr);
  if (packageModuleOr.isError())
    return packageModuleOr.takeError();

  // Strip the implicit package exports, we don't need these because we're going
  // to link the package into an existing module as-is.
  for (ExportInterface op : (*packageModuleOr)->getOps<ExportInterface>())
    if (op.isPackageExported())
      op.setNotExported();

  // Materialize all of the dependent packages now, making sure they also
  // get linked in properly.
  mlir::PassManager pm((*packageModuleOr)->getContext());
  pm.addPass(createMaterializePackages());
  if (failed(pm.run(**packageModuleOr)))
    return Error("unable to materialize dependent packages");

  return std::move(*packageModuleOr);
}

//===----------------------------------------------------------------------===//
// loadPreElaboratedBytecodeForLinking
//===----------------------------------------------------------------------===//

ErrorOr<DenseResourceElementsAttr> KGEN::loadPreElaboratedBytecodeForLinking(
    DenseResourceElementsAttr bytecodeAttr) {
  ErrorOr<OwningOpRef<ModuleOp>> packageModuleOr =
      loadPreElaboratedModuleForLinking(bytecodeAttr);
  if (packageModuleOr.isError())
    return packageModuleOr.takeError();

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  WriteableBufferRef str = WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(**packageModuleOr, *str)))
    return Error("could not write bytecode for package module");

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  return createResourceAttr((*packageModuleOr)->getContext(), str->getBuffer(),
                            "bytecode_" +
                                llvm::toHex(hash, /*LowerCase=*/true));
}
