//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "Cache/Support/Keys.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/CommonCLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"
#include <filesystem>

using namespace M;
using namespace LLCL;
using namespace Cache;

namespace {
/// This provides a zero-copy binary blob cache key struct. The idea is that it
/// should operate directly on Cache::BufferRef because that's what we use in
/// this tool, and it should be simple to read/write.
using BinaryBlobCacheKey = Keys::VariantTypeKey<Cache::BufferRef, StringRef>;

/// Describes an input file, or a cached object request. An input file is simply
/// a path, while a cached object request is `<hash> > <output-path>` or
/// `key:<key value> > <output-path>`. First format is useful if we are directly
/// specifying the hash, whereas the later is for cases where we need to derive
/// key after some transformation. This format could easily be extended to
/// include things like extra data to be included in the hash. The output path
/// can be `-`, in which case the output is written to `-o` (which itself could
/// be stdout or a file) with a newline after each object retrieved.
struct FileOrCachedObjectRequest {
  std::string hashOrFilename;
  std::string outFileName;

  std::filesystem::path getInputFilename() const { return {hashOrFilename}; }
  StringRef getHash() const { return hashOrFilename; }
  std::filesystem::path getOutputFilename() const { return {outFileName}; }
  bool isaGet() const {
    // When we don't have an output file name, it must be a PUT.
    return !outFileName.empty();
  }
};

/// Provide a parser for the FileOrCachedObjectRequest object.
class FileOrCachedObjectRequestParser
    : public llvm::cl::parser<FileOrCachedObjectRequest> {
public:
  using llvm::cl::parser<FileOrCachedObjectRequest>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             FileOrCachedObjectRequest &val);
};

/// Provides the CLOptions for this tool.
class CLOptions : public CLOptionsBase {
public:
  using CLOptionsBase::CLOptionsBase;

  /// Specify the input file or cached object.
  cl::list<FileOrCachedObjectRequest, bool, FileOrCachedObjectRequestParser>
      inputs{"input", cl::desc("<input file or CAS reference>"),
             llvm::cl::OneOrMore};

  cl::list<std::string> keys{
      "key",
      cl::desc(
          "Explicitly Specify key. In this case instead of binary hash, "
          "this key will be used for adding object to cache. Only valid for "
          "PUT operation. For get specify the key with --input option with a "
          "prefix 'key:'. For eg: key:my-key.")};

  cl::opt<std::string> target{
      "target",
      cl::desc(
          "Augment key with information specific to a target hardware (optional). \
           Accepted values: host, host-static, or a custom target of the format \
           arch:feature. e.g. x86_64:avx2 or x86_64:avx512f"),
      cl::init("none")};

  cl::opt<std::string> backendVersion{
      "backend-version", cl::desc("Set the version for the local backend."),
      cl::init("")};

  /// Specify the target path for the CAS backend.
  cl::opt<std::string> fsPath{
      "base-dir",
      cl::desc("Filesystem path for the CAS local storage. Defaults to a "
               "temporary directory."),
      llvm::cl::init("")};

  cl::opt<std::string> outFile{
      "o", cl::desc("File path to use for program outputs."),
      llvm::cl::init("-")};

  /// Get the filesystem path for the CAS. Defaults to a temporary directory.
  std::filesystem::path getFsPath() const;
};
} // namespace

static bool isCustomTargetValid(const llvm::StringRef &customTarget) {
  using namespace M::Cache::Keys;
  auto [uarch, feature] = customTarget.split(':');
  return !feature.empty() &&
         llvm::is_contained(CPUFeatureWrapper::SUPPORTED_ARCHS, uarch) &&
         llvm::is_contained(CPUFeatureWrapper::SUPPORTED_FEATURES, feature);
}

/// Wrap the given key with appropriate target info.
static std::string wrapKey(BinaryBlobCacheKey::KeyTy key,
                           const std::string &target) {
  if (target == "none")
    return BinaryBlobCacheKey::hashKey(std::move(key));
  if (target == "host")
    return Keys::KeyWithHostInfo<BinaryBlobCacheKey>::hashKey(std::move(key));
  if (target == "host-static")
    return Keys::KeyWithStaticHostInfo<BinaryBlobCacheKey>::hashKey(
        std::move(key));

  // `target` must be a custom target
  if (isCustomTargetValid(target)) {
    std::string hashedKey = BinaryBlobCacheKey::hashKey(std::move(key));
    return Keys::StringHashedKey::hashKey(hashedKey + target);
  }

  llvm::errs() << "[WARNING] Unable to understand custom target " << target
               << " defaulting to unwrapped key.";
  return BinaryBlobCacheKey::hashKey(std::move(key));
}

//===----------------------------------------------------------------------===//
// FileOrCachedObjectRequestParser::parse
//===----------------------------------------------------------------------===//

bool FileOrCachedObjectRequestParser::parse(llvm::cl::Option &o,
                                            StringRef argName,
                                            StringRef argValue,
                                            FileOrCachedObjectRequest &val) {
  // Check if the value contains the name and signature.
  SmallVector<StringRef> split;
  argValue.split(split, '>');
  // Trim whitespace from the ends of the string.
  llvm::for_each(split, [](StringRef &s) { s = s.ltrim().rtrim(); });
  if (split.size() == 2) {

    SmallVector<StringRef> tagSplit;

    split.front().split(tagSplit, ':');
    if (tagSplit.size() == 2) {
      // This is of the form tag:value
      val.hashOrFilename = tagSplit.back();
      val.hashOrFilename.shrink_to_fit();
    } else { // First try to get it as a hex string. If that fails, try base64.
      if (!llvm::tryGetFromHex(split[0], val.hashOrFilename)) {
        std::vector<char> ref;
        ref.reserve(split[0].size());
        if (auto err = llvm::decodeBase64(split[0], ref)) {
          o.error(toString(std::move(err)));
          return true;
        }
        val.hashOrFilename = std::string(ref.begin(), ref.end());
        val.hashOrFilename.shrink_to_fit();
      }
      if (val.hashOrFilename.empty() || val.hashOrFilename.size() != 32) {
        o.error("parsed hash could not be decoded into 32 bytes");
        return true;
      }
    }
    val.outFileName = split.back();
    return false;
  }

  // Make sure the arg value is not a directory.
  std::error_code ec;
  if (std::filesystem::is_directory(argValue.str(), ec)) {
    o.error(argValue + " is a directory, please redirect to a file");
    return true;
  }
  if (ec) {
    o.error(ec.message());
    return true;
  }

  // Otherwise, if it's not a hash, the value is the name.
  val.hashOrFilename = argValue;
  val.outFileName = "";
  return false;
}

//===----------------------------------------------------------------------===//
// CLOptions::getFsPath
//===----------------------------------------------------------------------===//

std::filesystem::path CLOptions::getFsPath() const {
  // Get the path provided to the command line if it exists.
  std::filesystem::path out(fsPath.getValue());
  std::error_code ec;
  if (!out.empty()) {
    out = std::filesystem::absolute(out, ec);
    if (ec) {
      reportError(ec.message());
      exit(1);
    }
    return out;
  }

  // Default to some temp directory.
  out = std::filesystem::temp_directory_path(ec) / "modular" / "cache";
  if (ec) {
    reportError(ec.message());
    exit(1);
  }
  llvm::errs() << "[WARNING] Using temporary file path at " << out.string()
               << " for CAS filesystem base path.\n";
  return out;
}

static AsyncValueRef<std::string>
putObjectsIntoCache(BinaryBlobCacheKey::KeyTy key, Cache::BufferRef value,
                    const FileOrCachedObjectRequest &input,
                    RCRef<BlobCache<BinaryBlobCacheKey>> &cache,
                    LLCL::Runtime &runtime) {

  AsyncValueRef<std::string> insert =
      cache->insert(std::move(key), std::move(value));
  auto outCh = AsyncValueRef<std::string>::allocate(runtime);
  std::move(insert).andThenSync(
      [outCh = outCh.copy(),
       input = Buffer::get(input.getInputFilename().string())](
          AsyncValueRef<std::string> &&hash) mutable {
        // If we have an error, report it.
        if (hash.isError())
          return std::move(outCh).setToError(hash.takeDiagnostic());

        // Otherwise, emplace the string so that we can report it to the
        // user.
        std::move(outCh).emplace(llvm::encodeBase64(*hash));
      });
  return outCh;
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  Runtime runtime(createLeakCheckAllocator(createMallocAllocator()),
                  createThreadPoolWorkQueue());

  auto backendChainOr = getLocalDefaultBackendChain(
      runtime, clOptions.getFsPath(), clOptions.backendVersion);
  if (backendChainOr.isError())
    return clOptions.reportError(backendChainOr.getError());

  auto cache =
      RCRef<BlobCache<BinaryBlobCacheKey>>::create(std::move(*backendChainOr));

  SmallVector<AnyAsyncValueRef> results;
  if (!clOptions.keys.empty()) {
    if (clOptions.inputs.size() != clOptions.keys.size())
      return clOptions.reportError(
          "Number of inputs should match number of keys.");

    for (const auto &[key, input] :
         llvm::zip(clOptions.keys, clOptions.inputs)) {
      auto bufOr = Cache::Buffer::getFile(input.getInputFilename());
      if (bufOr.isError())
        return clOptions.reportError(bufOr.getError());
      AsyncValueRef<std::string> outCh =
          putObjectsIntoCache(wrapKey(key, clOptions.target), (*bufOr).copy(),
                              input, cache, runtime);
      results.push_back(std::move(outCh));
    }
  }

  for (const FileOrCachedObjectRequest &input : clOptions.inputs) {
    // If it's a PUT, then hash the input and write it to the CAS.
    if (input.isaGet()) {
      // Attempt to find the value in the cache, if it exists, write it out.
      auto result = cache->find(wrapKey(input.getHash(), clOptions.target));
      auto outCh = AsyncValueRef<BufferRef>::allocate(runtime);
      std::move(result).andThenSync(
          [outCh = outCh.copy(), input = Buffer::get(input.getHash())](
              AsyncValueRef<std::optional<BufferRef>> &&found) mutable {
            // If there was an error, produce that diagnostic.
            if (found.isError())
              return std::move(outCh).setToError(found.takeDiagnostic());

            // No value, emit an error.
            if (!found->has_value()) {
              return std::move(outCh).setToError(
                  UnknownLocationDecoder::getDiagnostic(
                      Twine(llvm::encodeBase64(input->getBuffer())) +
                      ": value not found in the cache"));
            }

            // Has a value, write the value.
            BufferRef buf = std::move(**found);
            std::move(outCh).emplace(std::move(buf));
          });
      results.push_back(std::move(outCh));
    } else if (clOptions.keys.empty()) {
      auto bufOr = Cache::Buffer::getFile(input.getInputFilename());
      if (bufOr.isError())
        return clOptions.reportError(bufOr.getError());
      std::string key = wrapKey((*bufOr).copy(), clOptions.target);
      AsyncValueRef<std::string> outCh =
          putObjectsIntoCache(key, (*bufOr).copy(), input, cache, runtime);
      results.push_back(std::move(outCh));
    }
  }

  // Await for all the results to quiesce.
  await(results);

  std::string errMsg;
  std::unique_ptr<llvm::ToolOutputFile> outFile =
      mlir::openOutputFile(clOptions.outFile, &errMsg);
  if (!outFile)
    return clOptions.reportError(errMsg);

  // Report any errors we might have.
  for (auto [r, input] : llvm::zip(results, clOptions.inputs)) {
    if (r.isError())
      return clOptions.reportError(r.getDiagnostic().getMessage().get());
    // Emit a semicolon-separated list of hashes for the provided PUTs in order.
    if (r.isType<std::string>()) {
      outFile->os() << r.get<std::string>() << ";";
    } else if (r.isType<BufferRef>()) {
      std::filesystem::path outPath = input.getOutputFilename();
      StringRef buf = r.get<BufferRef>()->getBuffer();
      // Emit to stdout (or the file), so print it and carry on.
      if (outPath == "-") {
        outFile->os() << buf << "\n";
        continue;
      }

      // Otherwise, map a file and write the contents.
      auto writeableBuf = WriteableBuffer::getFile(outPath, buf.size());
      if (writeableBuf.isError())
        return clOptions.reportError(writeableBuf.getError());

      // Write the data to the mapped file.
      (*writeableBuf)->pwrite(buf.data(), buf.size(), 0);
    }
  }
  // Keep the output file.
  outFile->keep();
  return 0;
}
