//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_GUI_STRING_LIBRARY_H
#define MOTR_GUI_STRING_LIBRARY_H

#include <cstdint>
#include <deque>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace M::motr {

struct StringLibrary {
  using HashType = uint64_t;
  static constexpr size_t ChunkSize = 4 * 1024 * 1024;
  static constexpr size_t DefaultMaxChunks = 8;

  static_assert(ChunkSize * DefaultMaxChunks < (2 << 25),
                "StringLibrary takes more than 16 Megabytes");

  StringLibrary(size_t maxChunks = DefaultMaxChunks);

  // Add a single string to the library
  std::pair<std::string_view, HashType> addString2(std::string_view str);

  // Add a single string to the library
  std::string_view addString(std::string_view str);

  // Add multiple strings to the library
  std::vector<std::string_view>
  addStrings(const std::vector<std::string_view> &strs);

  // Get a single string from the library
  // if createPlaceholder is true, and the string is not found,
  // a placeholder in the form of "##<hex-hash>" will be created and returned
  // otherwise, an empty string_view will be returned
  std::string_view getString(uint64_t hash,
                             bool createPlaceholder = false) const;

  std::string_view getString(std::string_view str) const;

  std::string_view operator[](HashType hash) const;

  // Get multiple strings from the library
  std::vector<std::string_view>
  getStrings(const std::vector<uint64_t> &hashes,
             bool createPlaceholder = false) const;

  // Get a placeholder for a string from the library
  std::string_view getPlaceholder(uint64_t hash) const;

  // Get a placeholder for multiple strings from the library
  std::vector<std::string_view>
  getPlaceholders(const std::vector<uint64_t> &hashes) const;

  void setMaxChunks(size_t maxChunks);
  void evictChunksUntil(size_t maxChunks);

  struct Chunk;
  struct ChunkDeleter {
    void operator()(Chunk *chunk) const;
  };

  using ChunkPtr = std::unique_ptr<Chunk, ChunkDeleter>;

  void cleanupChunk(Chunk &chunk);

  size_t maxChunks;
  std::deque<ChunkPtr> chunks;
  ChunkPtr placeholderChunk;
  std::unordered_map<HashType, std::string_view> strings;
  size_t generation{};
};

} // namespace M::motr

#endif // MOTR_GUI_STRING_LIBRARY_H
