//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/StringLibrary.h"
#include "motr/Hash.h"
#include "motr/Log.h"
#include <cassert>

using namespace M::motr;

StringLibrary::Chunk::Chunk() {
  static size_t static_generation = 0;
  generation = static_generation++;
  //  MOTR_LOG("StringLibrary::Chunk({}) created", generation);
}

StringLibrary::Chunk::~Chunk() {
  MOTR_LOG("StringLibrary::Chunk({}) destroyed with {} strings and {} bytes, "
           "{} unused ({:4.1f}%)",
           generation, strings.size(), offset, (Size - offset),
           (Size - offset) * 100.0 / Size);
}

std::string_view StringLibrary::Chunk::getString(HashType hash) const {
  auto it = strings.find(hash);
  if (it != strings.end())
    return it->second;
  return {};
}

bool StringLibrary::Chunk::hasString(HashType hash) const {
  return strings.find(hash) != strings.end();
}

std::string_view StringLibrary::Chunk::copyToArena(std::string_view str) {
  size_t datasize = str.size() + 1;
  assert(datasize <= Size && "String is too large to fit in any Chunk");

  // string is too large to fit in THIS chunk
  if (offset + datasize > Size)
    return {};

  // copy the string into the chunk
  auto *arenaStrBegin = arena.data() + offset;
  std::copy(str.begin(), str.end(), arenaStrBegin);
  offset += str.size();
  arena[offset] = '\0';
  offset += 1;

  return {arenaStrBegin, str.size()};
}

std::string_view StringLibrary::Chunk::addString(HashType hash,
                                                 std::string_view instr) {
  // this assert is disabled because the hash MAY be different
  // because this chunk is a placeholder chunk
  // which maps unfound hashes to a placeholder string
  // that actually has a different hash
  // assert(Hash::of(str) == hash && "String hash mismatch");

  std::string_view sv = getString(hash);
  if (sv.data() != nullptr)
    return sv;

  sv = copyToArena(instr);

  // This chunk cannot hold instr
  if (sv.data() == nullptr)
    return {};

  strings[hash] = sv;
  assert(isStringViewMemoryInArena(sv) &&
         "sanity check failure: String is not in arena");
  return sv;
}

StringLibrary::StringLibrary(size_t maxChunks)
    : maxChunks(maxChunks), // use setMaxChunks() to change
      placeholderChunk(std::make_unique<Chunk>()), //
      strings({})                                  //
{
  chunks.emplace_back(std::make_unique<Chunk>());
}

void StringLibrary::setMaxChunks(size_t newMaxChunks) {
  maxChunks = newMaxChunks;
  evictChunksUntil(maxChunks);
}

std::string_view StringLibrary::getPlaceholder(uint64_t hash) const {
  std::string_view sv = placeholderChunk->getString(hash);
  if (sv.data() != nullptr)
    return sv;
  sv = placeholderChunk->addString(hash, fmt::format("##{:016x}", hash));
  return sv;
}

std::string_view StringLibrary::getString(uint64_t hash,
                                          bool createPlaceholder) const {
  if (auto it = strings.find(hash); it != strings.end())
    return it->second;

  if (createPlaceholder)
    return getPlaceholder(hash);

  return {};
}

std::vector<std::string_view>
StringLibrary::getStrings(const std::vector<uint64_t> &hashes,
                          bool createPlaceholder) const {
  std::vector<std::string_view> views;
  views.reserve(hashes.size());
  for (auto hash : hashes)
    views.push_back(getString(hash, createPlaceholder));

  return views;
}

inline bool
StringLibrary::Chunk::isStringViewMemoryInArena(std::string_view sv) const {
  auto *arenaStart = arena.data();
  auto *arenaEnd = arenaStart + Size + 1;
  auto *stringStart = sv.data();
  auto *stringEnd = stringStart + sv.size() + 1;
  bool isStartInArena = stringStart >= arenaStart && stringStart < arenaEnd;
  bool isEndInArena = stringEnd >= arenaStart && stringEnd < arenaEnd;
  return isStartInArena && isEndInArena;
}

// removes all strings from the library that are in the given chunk
void StringLibrary::cleanupChunk(Chunk &chunk) {
  MOTR_LOG("cleanupChunk({}) has {} strings", chunk.generation,
           chunk.strings.size());
  for (const auto &[hash, str] : chunk.strings) {
    auto it = strings.find(hash);
    assert(it != strings.end() && "Chunk string not found in library");
    if (it != strings.end()) {
      std::string_view library_sv = it->second;
      bool library_sv_in_arena = chunk.isStringViewMemoryInArena(library_sv);
      assert(library_sv_in_arena && "String is not in arena");
      assert(std::string(it->second) == std::string(str) && "String mismatch");
      strings.erase(it);
    }
  }
}

void StringLibrary::evictChunksUntil(size_t desiredChunks) {
  assert(desiredChunks <= maxChunks &&
         "Desired chunks is greater than the number of chunks");
  assert(desiredChunks > 0 &&
         "StringLibrary cannot be empty (desiredChunks is 0)");

  while (chunks.size() > desiredChunks) {
    MOTR_LOG("StringLibrary::Chunk (Size={}) evicting first of {} to reach "
             "maxChunks={}",
             Chunk::Size, chunks.size(), maxChunks);
    cleanupChunk(*chunks.front());
    chunks.pop_front();
  }
}

// adds a single string to the library
// the single add case is where the logic is implemented
// as there are no efficiencies when adding multiple strings
// because each string needs to be individually checked for
// whether it fits in a chunk or not
std::string_view StringLibrary::addString(std::string_view str) {
  size_t size = str.size() + 1;
  size_t datasize = str.size() + 1;

  assert(datasize <= Chunk::Size && "String is too large to fit in a chunk");

  // check if the string is a placeholder
  // if you're adding it here, that means you're using the API wrong
  if (str.size() == 18 && str[0] == '#' && str[1] == '#') {
    MOTR_LOG("String is a placeholder: {}", str);
    assert(false && "String is a placeholder");
    return {};
  }

  uint64_t hash = Hash::of(str);

  std::string_view sv = getString(hash);
  if (sv.data() != nullptr)
    return sv;

  assert(!chunks.empty() && "No chunks in library");

  sv = chunks.back()->addString(hash, str);
  // if the string could not be added to the last chunk,
  // we need to create a new chunk
  // We could possibly insert the string into older chunks
  // that may still have space available.
  // But that would then make the string in an "old" chunk
  // and result in early eviction of the string.
  if (sv.data() == nullptr) {
    // ensure at least one chunk can be allocated
    evictChunksUntil(maxChunks - 1);
    chunks.push_back(std::make_unique<Chunk>());
    sv = chunks.back()->addString(hash, str);
  }

  assert(sv.data() != nullptr && "String could not be added to any chunk");

  if (sv.data() != nullptr) {
    // string view was successfully stashed in the library
    strings[hash] = sv;
    // increment the generation number
    generation++;
  }

  return sv;
}

std::string_view StringLibrary::operator[](HashType hash) const {
  return getString(hash, true);
}

// adds multiple strings to the library
std::vector<std::string_view>
StringLibrary::addStrings(const std::vector<std::string_view> &strs) {
  std::vector<std::string_view> views;
  views.reserve(strs.size());

  for (auto str : strs)
    views.push_back(addString(str));

  return views;
}
