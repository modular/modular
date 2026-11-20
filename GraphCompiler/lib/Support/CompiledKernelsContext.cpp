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

#include "GraphCompiler/Support/CompiledKernelsContext.h"
#include "GraphCompiler/Support/List.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "Mojo/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "Support/AssertStream.h"
#include "Support/FileSystemExtras.h"
#include "Support/Preprocessor.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "compiled-kernels-context"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace M::GML;

//===----------------------------------------------------------------------===//
// CompiledKernelCall
//===----------------------------------------------------------------------===//

ErrorOrSuccess CompiledKernelABI::CompiledKernelCall::pushObject(
    AnyAsyncValueRef input, char inOutType, StringRef summary,
    const EncodedLocation &loc) {
  if (input.isType<TensorBufferRef>()) {
    const void *ptr = input.get<TensorBufferRef>().getBuffer();
    // Forget the const here since the ABI mixes input and output.
    inputPointers.push_back(const_cast<void *>(ptr));

  } else if (input.isType<GML::List>()) {
    auto &runtimeList = input.get<GML::List>();
    auto &abiTensors = tensorsMemory.emplace_back();

    // NOTE: Not closed under all argument types, and not extensible by
    // subclasses.
    for (GML::Variant &ref : runtimeList.getElements()) {
      if (ref.getValue().isType<Tensor>()) {
        auto &runtimeTensor = ref.getValue().get<Tensor>();
        const TensorShape &shape = runtimeTensor.getShape();
        ssize_t *dims;
        // Mojo needs scalars as 1D.
        if (shape.getRank() == 0)
          dims = dimsMemory.emplace_back(1, 1).data();
        else
          dims = dimsMemory.emplace_back(shape.begin(), shape.end()).data();
        void *data = const_cast<void *>(runtimeTensor.getBuffer());
        abiTensors.emplace_back(dims, data);
      } else {
        return Error(Twine("Invalid list element type: ") +
                     ref.getValue().getPointer()->getTypeID().getTypeName());
      }
    }

    auto &abiListPtr = listMemory.emplace_back(std::make_unique<ABIList>(
        runtimeList.getElements().size(), abiTensors.data()));
    inputPointers.push_back(abiListPtr.get());

  } else {
    return Error(Twine("Invalid input type: ") +
                 input.getPointer()->getTypeID().getTypeName());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CompiledKernelsContext
//===----------------------------------------------------------------------===//

ErrorOrSuccess CompiledKernelsContext::load(ArrayRef<uint8_t> binary,
                                            ArrayRef<StringRef> kernelNames,
                                            size_t kernelIndex) {
  // Kernels from different models may share names, and one model may be set up
  // more than once. So load into a dylib named by the archive hash, created
  // exactly once via the engine's atomic 'addIfAbsent'.
  std::string dylibName = llvm::utohexstr(llvm::xxh3_64bits(binary),
                                          /*LowerCase=*/true, /*Width=*/16);
  auto buf = Buffer::get(
      StringRef(reinterpret_cast<const char *>(binary.data()), binary.size()));

  // With debuginfo, link the kernel objects into a shared library and load it
  // back so profilers can see the symbols after this program exits.
  if (loadDebuginfo) {
    auto dylibOr = engine->getExecutionSession().createJITDylib(dylibName);
    if (!dylibOr)
      return Error("Could not create JIT dylib");
    llvm::orc::JITDylib &dylib = *dylibOr;

    auto objectBinaryOr =
        toModularErrorOr(llvm::object::createBinary(buf->getMemBufferRef()));
    if (objectBinaryOr.isError())
      return objectBinaryOr.takeError();
    llvm::Error err = llvm::Error::success();
    std::unique_ptr<llvm::object::Binary> objectBinary =
        objectBinaryOr.takeValue();

    std::string libName =
        std::filesystem::path(dylibName).filename().string() + "_" +
        std::filesystem::path(objectBinary->getFileName().str())
            .stem()
            .string();

    auto objectBuffer = objectBinary->getMemoryBufferRef();
    auto tmpFile = writeTempFile(libName + "-%%%%.o", objectBuffer.getBuffer());

    if (tmpFile.isError())
      return M::Error(std::string(tmpFile.takeError().get()) + ": " + libName);
    // On MacOS debuginfo is stored in object files and referenced by shared
    // libraries. Keep the object files around so the debuginfo can be found.
    tmpFile->keep();
    std::string objectFile = tmpFile->getPath();
    tmpFile->close();

    if (err)
      return toModularError(std::move(err));

    SmallString<0> outPath;
    std::error_code ec = llvm::sys::fs::createUniqueFile(
        (std::filesystem::temp_directory_path() / dylibName).string() +
            "-%%%%.so",
        outPath);
    if (ec)
      return Error("Could not create tmp file: " + ec.message());
    // Kernels only reference compiler_rt.a; resolve those symbols by dynamic
    // lookup instead of linking it.
    // TODO: run linker programmatically
    std::string cmd;
    llvm::raw_string_ostream cmdS(cmd);
    cmdS << std::string(GENERICML_LINKER_CMD) << " -shared ";
    cmdS << objectFile;
    cmdS << " -o " << outPath;
    LLVM_DEBUG(llvm::dbgs() << "Linked shared library: " << cmd << "\n");
    int r = system(cmd.c_str());
    if (r != 0)
      return Error("Could not link shared object");
    LLVM_DEBUG(llvm::dbgs() << "Loading shared library " << outPath << "\n");
    auto gen = llvm::orc::DynamicLibrarySearchGenerator::Load(
        outPath.c_str(), engine->getDataLayout().getGlobalPrefix());
    if (!gen)
      return M::Error(toModularError(gen.takeError()));
    dylib.addGenerator(std::move(gen.get()));
    if (auto err = engine->addToSearchOrder(dylibName, &dylib))
      return err.takeError();
  } else {
    if (auto err = engine->addIfAbsent<KGEN::StaticArchiveLayer>(
            dylibName, std::move(buf)))
      return err.takeError();
  }

  for (StringRef kernelName : kernelNames) {
    auto symOr = engine->lookup(dylibName, kernelName);
    if (symOr.isError())
      return symOr.takeError();
    addKernel(kernelName, kernelIndex, symOr.takeValue().getFunctionPointer());
    ++kernelIndex;
  }

  return success();
}

void CompiledKernelsContext::addKernel(StringRef name, size_t index,
                                       void *kernel) {
  ASSERT_STREAM(
      kernels.size() == index,
      << "All kernels must be added to symbol table at the same time.");
  kernels.push_back(kernel);
  symbolTable.push_back(name);
}
