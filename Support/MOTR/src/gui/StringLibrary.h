//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_GUI_STRING_LIBRARY_H
#define MOTR_GUI_STRING_LIBRARY_H

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace M::motr {

struct StringLibrary {
  using HashType = uint64_t;
  static constexpr size_t ChunkSize = 4 * 1024 * 1024;
  static constexpr size_t DefaultMaxChunks = 8;

  static_assert(ChunkSize * DefaultMaxChunks < (2 << 25),
                "StringLibrary takes more than 16 Megabytes");

  struct Chunk {
    static constexpr size_t Size = StringLibrary::ChunkSize;
    std::array<char, Size> arena;
    size_t generation{};
    size_t offset{};
    std::unordered_map<HashType, std::string_view> strings;
    Chunk();
    ~Chunk();
    // allow move, disallow copy
    Chunk(const Chunk &) = delete;
    Chunk &operator=(const Chunk &) = delete;
    Chunk(Chunk &&) = default;
    Chunk &operator=(Chunk &&) = default;

    std::string_view getString(HashType hash) const;
    bool hasString(HashType hash) const;
    std::string_view copyToArena(std::string_view str);
    std::string_view addString(HashType hash, std::string_view str);
    bool isStringViewMemoryInArena(std::string_view sv) const;
  };

  // Constructor
  StringLibrary(size_t maxChunks = DefaultMaxChunks);

  // Add a single string to the library
  std::string_view addString(std::string_view str);

  // Add multiple strings to the library
  std::vector<std::string_view>
  addStrings(const std::vector<std::string_view> &strs);

  // Get a single string from the library
  std::string_view getString(uint64_t hash,
                             bool createPlaceholder = false) const;

  // Get multiple strings from the library
  std::vector<std::string_view>
  getStrings(const std::vector<uint64_t> &hashes,
             bool createPlaceholder = false) const;

  // Get a placeholder for a string from the library
  std::string_view getPlaceholder(uint64_t hash) const;

  // Get a placeholder for multiple strings from the library
  std::vector<std::string_view>
  getPlaceholders(const std::vector<uint64_t> &hashes) const;

  // equivalent to getString({hash}, true)
  std::string_view operator[](HashType) const;

  void setMaxChunks(size_t maxChunks);
  void evictChunksUntil(size_t maxChunks);
  void cleanupChunk(Chunk &chunk);

  size_t maxChunks;
  std::deque<std::unique_ptr<Chunk>> chunks;
  std::unique_ptr<Chunk> placeholderChunk;
  std::unordered_map<HashType, std::string_view> strings;

  // generation counter allows consumers of the StringLibrary
  // to detect inserts and optionally re-query the library
  // hashes that were recently added to the library
  size_t generation{};
};

} // namespace M::motr

#endif // MOTR_GUI_STRING_LIBRARY_H
