//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/CommonCLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Regex.h"
#include <filesystem>

using namespace M;
using namespace LLCL;
using namespace Cache;

namespace {
/// This provides a zero-copy binary blob cache key struct. The idea is that it
/// should operate directly on Cache::BufferRef because that's what we use in
/// this tool, and it should be simple to read/write.
struct BinaryBlobCacheKey {
  using KeyTy = std::variant<Cache::BufferRef, StringRef>;
  static std::string hashKey(KeyTy key) {
    if (std::holds_alternative<StringRef>(key))
      return std::get<StringRef>(key).str();

    auto &bytes = std::get<Cache::BufferRef>(key);

    llvm::BLAKE3 hashState;
    hashState.update(bytes->getBuffer());
    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};

/// Describes an input file, or a cached object request. An input file is simply
/// a path, while a cached object request is `<hash>:<output-path>`. This format
/// could easily be extended to include things like extra data to be included in
/// the hash.
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

  /// Specify the target path for the CAS backend.
  cl::opt<std::string> fsPath{
      "base-dir",
      cl::desc("Filesystem path for the CAS local storage. Defaults to a "
               "temporary directory."),
      llvm::cl::init("")};

  /// Get the filesystem path for the CAS. Defaults to a temporary directory.
  std::filesystem::path getFsPath() const;
};
} // namespace

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
    // First try to get it as a hex string. If that fails, try base64.
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

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  Runtime runtime(createLeakCheckAllocator(createMallocAllocator()),
                  createThreadPoolWorkQueue());

  auto backendChainOr = getDefaultBackendChain(runtime, clOptions.getFsPath());
  if (backendChainOr.isError())
    return clOptions.reportError(backendChainOr.getError());

  auto cache =
      RCRef<BlobCache<BinaryBlobCacheKey>>::create(std::move(*backendChainOr));

  SmallVector<AnyAsyncValueRef> results;
  for (const FileOrCachedObjectRequest &input : clOptions.inputs) {
    // If it's a PUT, then hash the input and write it to the CAS.
    if (input.isaGet()) {
      // Attempt to find the value in the cache, if it exists, write it out.
      auto result = cache->find(input.getHash());
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
    } else {
      auto bufOr = Cache::Buffer::getFile(input.getInputFilename());
      if (bufOr.isError())
        return clOptions.reportError(bufOr.getError());

      auto insert = cache->insert((*bufOr).copy(), (*bufOr).copy());
      auto outCh = AsyncValueRef<std::string>::allocate(runtime);
      std::move(insert).andThenSync(
          [outCh = outCh.copy(),
           input = Buffer::get(input.getInputFilename().string())](
              AsyncValueRef<ErrorOr<std::string>> &&hash) mutable {
            // If we have an error, report it.
            if (hash.isError())
              return std::move(outCh).setToError(hash.takeDiagnostic());
            if (hash->isError())
              return std::move(outCh).setToError(
                  UnknownLocationDecoder::getDiagnostic(hash->takeError()));

            // Otherwise, emplace the string so that we can report it to the
            // user.
            std::move(outCh).emplace(llvm::encodeBase64(**hash));
          });
      results.push_back(std::move(outCh));
    }
  }
  // Await for all the results to quiesce.
  await(results);
  // Report any errors we might have.
  for (auto [r, input] : llvm::zip(results, clOptions.inputs)) {
    if (r.isError())
      return clOptions.reportError(r.getDiagnostic().getMessage().get());
    // Emit a semicolon-separated list of hashes for the provided PUTs in order.
    if (r.isType<std::string>()) {
      llvm::outs() << r.get<std::string>() << ";";
    } else if (r.isType<BufferRef>()) {
      std::filesystem::path outPath = input.getOutputFilename();
      StringRef buf = r.get<BufferRef>()->getBuffer();
      // Emit to stdout, so print it and carry on.
      if (outPath == "-") {
        llvm::outs() << buf << "\n";
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
  return 0;
}
