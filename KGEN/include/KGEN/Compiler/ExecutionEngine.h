//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_EXECUTIONENGINE_H
#define KGEN_COMPILER_EXECUTIONENGINE_H

#include "Support/Buffer.h"
#include "Support/Compiler/Sanitizers.h"
#include "Support/ErrorOr.h"
#include "Support/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/TypeName.h"

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace M::KGEN {
//===----------------------------------------------------------------------===//
// ExecutionEngineOptions
//===----------------------------------------------------------------------===//

/// This is a struct of options that the ExecutionEngine wants to have on
/// construction. These are like the KGEN compilation options, but we want to
/// avoid depending on them directly.
struct ExecutionEngineOptions {
  /// Whether or not to register the GDB plugins.
  bool registerDebugPlugins = false;
  bool registerPerfPlugins = false;

  /// Set to true if the executing engine is being used to cross-compile. This
  /// will forgo any JIT setup and capabilities.
  bool crossCompiling = false;

  /// An ORC ExecutorProcessControl that the user can specify.
  std::unique_ptr<llvm::orc::ExecutorProcessControl> epc = nullptr;
};

//===----------------------------------------------------------------------===//
// CompiledFunc
//===----------------------------------------------------------------------===//

/// This class provides an interface to interact with a compiled func. You
/// can either invoke the func, or get it as an object. The lifetime of one of
/// these objects is tied to the ExecutionEngine through the `cache` member.
/// This could be relaxed by using a pointer instead, but that would require
/// getObject to fail if the cache is unavailable, and there's currently no use
/// case for such a feature so we will leave it to the future.
class CompiledFunc {
public:
  /// Invoke this func. This has exactly the signature the compiled func
  /// does. Intended to have perfect forwarding of arguments into the
  /// function, and of return values from the function.
  template <typename ReturnT, typename... Args>
  ReturnT invoke(Args... args) {
    // Cast the function pointer and invoke it directly.
    return ((ReturnT(*)(Args...))fn)(std::forward<Args>(args)...);
  }

  /// Return the pointer to the compiled function.
  void *getFunctionPointer() const { return fn; }

private:
  /// Construct a CompiledFunc object. This constructor is private because it
  /// needs a reference to the cache that the ExecutionEngine holds, so it
  /// should really only be constructed from the ExecutionEngine or something
  /// like it.
  CompiledFunc(void *ptr) : fn(ptr) {}
  friend class ExecutionEngine;

  /// Pointer to the function to invoke.
  void *fn;
};

//===----------------------------------------------------------------------===//
// MaterializationLayer
//===----------------------------------------------------------------------===//

/// Provides a base class we can use to store pointers to Layers in the
/// ExecutionEngine.
///
/// Layers must implement an `add` function that the ExecutionEngine can use:
///
///  ErrorOrSuccess add(StringRef libName, Args args...);
///
/// The base class doesn't have a virtual method to override largely because
/// each class will have its own requirements for what needs to be passed into
/// `add`.
class MaterializationLayer {
public:
  enum LayerKind {
    kStaticSymbolLayer,
    kStaticArchiveLayer,
    kObjectCompilerLayer,
    kKGENCompilerLayer
  };
  virtual ~MaterializationLayer() = default;

  /// Nothing in this class hierarchy is copyable.
  MaterializationLayer(const MaterializationLayer &other) = delete;

  /// Check if this layer has an error.
  bool hasError() const { return error.has_value(); }

  LayerKind getKind() const { return kind; }

  /// Take the error from this layer.
  Error takeError() {
    assert(hasError());
    return std::move(*error);
  }

protected:
  using AddToSearchOrderFn =
      llvm::unique_function<ErrorOrSuccess(StringRef, llvm::orc::JITDylib *)>;

  MaterializationLayer(LayerKind kind, llvm::orc::ExecutionSession &sess,
                       const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Get or create a dylib with name `libName`. Subclasses should always use
  /// this method rather than manipulating the ExecutionSession directly.
  ErrorOr<llvm::orc::JITDylib *> getOrCreateDylib(StringRef libName);

  /// Mangle and intern `name` in the ExecutionSession.
  llvm::orc::SymbolStringPtr mangleAndIntern(StringRef name);

  /// Layers should override this function if they need to filter the symbols
  /// coming from the current process. The MaterializationLayer automatically
  /// adds visiblity to current process symbols when creating a new dylib, so
  /// this allows layers to customize that behavior.
  virtual llvm::unique_function<bool(const llvm::orc::SymbolStringPtr &)>
  getTargetProcessSymbolFilter() {
    return {};
  }

protected:
  llvm::orc::ExecutionSession &session;
  const llvm::DataLayout &dataLayout;
  AddToSearchOrderFn addToSearchOrder;

  /// Stores an optional Error that an individual layer can set to be checked
  /// later. This is necessary because the MaterializationUnit may call into a
  /// function in the layer that has no other way to report that error.
  std::optional<Error> error = std::nullopt;

private:
  LayerKind kind;
};

//===----------------------------------------------------------------------===//
// StaticSymbolLayer
//===----------------------------------------------------------------------===//

/// This layer provides a way to add a static symbol to the ExecutionEngine. The
/// symbol must have a static name (which is mangled for you) and resolve to an
/// address within the binary.
class StaticSymbolLayer : public MaterializationLayer {
public:
  /// The StaticSymbolLayer doesn't require anything extra to construct, just
  /// the `MaterializationLayer` arguments.
  StaticSymbolLayer(llvm::orc::ExecutionSession &sess,
                    const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add a function named `funcName` with address `fn` to the library
  /// `libName`.
  ErrorOrSuccess add(StringRef libName, StringRef funcName, void *fn);

  static bool classof(const MaterializationLayer *layer) {
    return layer->getKind() == LayerKind::kStaticSymbolLayer;
  }
};

//===----------------------------------------------------------------------===//
// StaticArchiveLayer
//===----------------------------------------------------------------------===//

/// This layer provides a way to add a static archive to the ExecutionEngine.
/// All symbols in the archive are made available for use and lookup.
class StaticArchiveLayer : public MaterializationLayer {
public:
  /// The StaticArchiveLayer needs a reference to the base object linking layer
  /// so it can feed the archive bytes into the linker.
  StaticArchiveLayer(llvm::orc::ObjectLayer &objLayer,
                     llvm::orc::ExecutionSession &sess,
                     const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add the archive in `archive` to the library `libName`. Stores a reference
  /// to `archive` inside the class to ensure its lifetime matches the lifetime
  /// of the ExecutionEngine.
  ErrorOrSuccess add(StringRef libName, BufferRef archive);

  static bool classof(const MaterializationLayer *layer) {
    return layer->getKind() == LayerKind::kStaticArchiveLayer;
  }

private:
  llvm::orc::ObjectLayer &objectLayer;
  SmallVector<BufferRef> archiveBuffers;
};

//===----------------------------------------------------------------------===//
// ExecutionEngine
//===----------------------------------------------------------------------===//

/// This class provides an interface to the LLVM ORCJIT. It can compile
/// individual funcs (already lowered to the LLVM dialect) to object code. It
/// caches the objects themselves so we can retrieve them later and write them
/// to a file. The fundamental unit this class deals with is a single llvm
/// function because that's the minimum granularity we would want to use for
/// caching and search.
class ExecutionEngine {
public:
  ~ExecutionEngine();

  //===--------------------------------------------------------------------===//
  // Constructors
  //===--------------------------------------------------------------------===//

  /// Create an ExecutionEngine with no layers. This is generally not very
  /// useful unless the user wants to customize exactly which layers go into the
  /// JIT.
  static ErrorOr<std::unique_ptr<ExecutionEngine>>
  create(ExecutionEngineOptions options, const llvm::TargetMachine &tm);

  /// Create an ExecutionEngine with the 2 standard layers: StaticSymbolLayer
  /// and StaticArchiveLayer. These two enable the user to expose
  /// current-process symbols into the JIT and load static archives, which is
  /// pretty much the base requirement.
  static ErrorOr<std::unique_ptr<ExecutionEngine>>
  createWithStandardLayers(ExecutionEngineOptions options,
                           const llvm::TargetMachine &tm);

  //===--------------------------------------------------------------------===//
  // Adding/finding layers
  //===--------------------------------------------------------------------===//

  /// Add a layer into the ExecutionEngine. This passes the variables that are
  /// private to the ExecutionEngine into the layer constructor and constructs
  /// the layer in-place.
  template <typename T, typename... Args>
  T &addLayer(Args &&...args) {
    assert(findLayer<T>() == nullptr && "duplicate layer found");
    return cast<T>(*layers.emplace_back(std::make_unique<T>(
        std::forward<Args>(args)..., *executionSession, dataLayout,
        [&](StringRef name, llvm::orc::JITDylib *dylib) {
          return addToSearchOrder(name, dylib);
        })));
  }

  /// Find a layer of type T. Because we can only have one layer of each kind,
  /// this simply iterates the layer list and returns a stable pointer to the
  /// layer of type T. If that layer cannot be found, returns nullptr.
  template <typename T>
  T *findLayer() const {
    auto found =
        llvm::find_if(layers, [](const auto &layer) { return isa<T>(*layer); });
    if (found == layers.end())
      return nullptr;
    return &cast<T>(**found);
  }

  /// Get a layer of type T. Asserts that the layer is found in the
  /// ExecutionEngine and returns a reference to it.
  template <typename T>
  T &getLayer() const {
    auto found =
        llvm::find_if(layers, [](const auto &layer) { return isa<T>(*layer); });
    assert(found != layers.end() && "can't find this layer...");
    return cast<T>(**found);
  }

  //===--------------------------------------------------------------------===//
  // Adding symbols/objects/etc.
  //===--------------------------------------------------------------------===//

  /// Add *something* to the ExecutionEngine. Uses `LayerT` to find the layer to
  /// add *to*, and then calls the layer's `add` function.
  template <typename LayerT, typename... Args>
  ErrorOrSuccess add(StringRef libName, Args &&...args) {
    LayerT *found = findLayer<LayerT>();
    if (!found)
      return Error("could not find layer of type " +
                   llvm::getTypeName<LayerT>());

    return found->add(libName, std::forward<Args &&>(args)...);
  }

  /// Constructs and adds an object with libName to the layer of LayerT.
  /// However, if libName already exists then is a no-op. Thread safe.
  template <typename LayerT, typename... Args>
  ErrorOrSuccess addIfAbsent(StringRef libName, Args &&...args) {
    LayerT *found = findLayer<LayerT>();
    if (!found)
      return Error("could not find layer of type " +
                   llvm::getTypeName<LayerT>());

    std::lock_guard<std::mutex> guard(mu);
    if (executionSession->getJITDylibByName(libName))
      return success();

    return found->add(libName, std::forward<Args &&>(args)...);
  }

  llvm::orc::ExecutionSession &getExecutionSession() {
    return *executionSession;
  }

  //===--------------------------------------------------------------------===//
  // Compiled symbol lookup
  //===--------------------------------------------------------------------===//

  /// Look up a func and return it as a CompiledFunc object if we can find it.
  ErrorOr<CompiledFunc> lookup(StringRef symbol);

  /// Look up the provided symbol only in the provided dylib and any others
  /// added to its link order. Note that this bypasses the default search order,
  /// and must therefore must be used with caution.
  ErrorOr<CompiledFunc> lookup(StringRef libName, StringRef symbol);

  //===--------------------------------------------------------------------===//
  // JIT Execution
  //===--------------------------------------------------------------------===//

  /// Run the entry point in the specified library as the main function of a
  /// program. This will invoke the entry point through the ORC RT if available.
  ErrorOrSuccess runProgram(StringRef libName, StringRef entryPoint,
                            function_ref<ErrorOrSuccess(void *)> runFn);

  /// Get the name of the global constructor function to call in JIT mode.
  static constexpr const char *getGlobalCtorFnName() {
    return "KGEN_EE_JIT_GlobalConstructor";
  }
  /// Get the name of the global destructor function to call in JIT mode.
  static constexpr const char *getGlobalDtorFnName() {
    return "KGEN_EE_JIT_GlobalDestructor";
  }

  //===--------------------------------------------------------------------===//
  // Misc
  //===--------------------------------------------------------------------===//

  /// Get the base object linking layer.
  llvm::orc::ObjectLinkingLayer &getLinkingLayer() { return *objectLayer; }
  const llvm::DataLayout &getDataLayout() const { return dataLayout; }

  /// Add a JITDylib to the search order for symbol resolution. Asserts if the
  /// dylib already exists - users should generally be cautious about adding
  /// dylibs to the search order.
  ErrorOrSuccess addToSearchOrder(StringRef name, llvm::orc::JITDylib *dylib);

private:
  explicit ExecutionEngine(std::unique_ptr<llvm::orc::ExecutionSession> session,
                           const llvm::DataLayout &dl);

  /// This class is not copy-constructible.
  ExecutionEngine(const ExecutionEngine &other) = delete;

  /// Mangle and intern a string name.
  llvm::orc::SymbolStringPtr mangleAndIntern(StringRef name);

  /// Look up the provided symbol with the given search order. This is a
  /// generalization of the two lookup methods above, we just don't want to
  /// expose the notion of a 'search order' to users cause it's easy to mis-use.
  ErrorOr<CompiledFunc>
  lookupWithSearchOrder(const llvm::orc::JITDylibSearchOrder &order,
                        StringRef symbol);

  /// The ORC requires an ExecutionSession - this is how it coordinates
  /// execution across processes/machines.
  std::unique_ptr<llvm::orc::ExecutionSession> executionSession = nullptr;

  /// JITLink linker. This is what drives all the linking underneath our JIT.
  std::unique_ptr<llvm::orc::ObjectLinkingLayer> objectLayer = nullptr;

  /// Protects the addition to all the layers in 'layers' when called via
  /// the thread-safe addIfAbsent.
  std::mutex mu;

  /// List of materialization layers the JIT has. The base is *always* the
  /// object linking layer. We are not likely to have more than 5 layers total:
  /// (1) StaticSymbolLayer, (2) StaticArchiveLayer, (3) KGEN object generation,
  /// (4) KGEN compilation pipeline, and (5) mojo parsing.
  SmallVector<std::unique_ptr<MaterializationLayer>, 5> layers;

  /// Keep a set of known dylibs and a dylib search order - this will make it
  /// easy to (a) make sure we only have unique dylibs and (b) cache the
  /// search order so we don't recreate it on every lookup.
  llvm::StringSet<> knownDylibs;
  llvm::orc::JITDylibSearchOrder searchOrder;

  /// We need to hold onto a pointer to the data layout because it holds onto
  /// some state.
  llvm::DataLayout dataLayout;

  /// List of buffers that contain archive files added to the JIT. This holds
  /// references to them so they aren't deallocated underneath our feet.
  SmallVector<BufferRef> archiveBuffers;
};
} // namespace M::KGEN

#endif // KGEN_COMPILER_EXECUTIONENGINE_H
