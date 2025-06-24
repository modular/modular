//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_FLAGS_H
#define MOTR_FLAGS_H

#include "motr/Hash.h"
#include "motr/Macros.h"
#include "motr/Namespace.h"
#include "motr/SharedMemory.h"
#include "motr/Time.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>

#define ALWAYS_INLINE __attribute__((always_inline))

namespace M::motr::Flags {
template <const char *name, typename T, typename = void>
struct FlagT;
struct Manager;
struct Flag;

namespace detail {

MOTR_ALWAYS_INLINE std::string getSharedMemoryName() {
  return Namespace::makeSHMName("flags");
}

// This value is used to indicate that the flag has not been initialized
// if you set a flag to this value, it is possible for someone else to
// accidentally read the flag initialize it to another value
// so be careful.  This is a hack to avoid having to add a new column to the
// shared memory table to indicate that the flag has not been initialized.
static constexpr const uint64_t UNINITIALIZED_ATOMIC_FLAG_VALUE =
    10101010101010101010ULL;
} // namespace detail

} // namespace M::motr::Flags

namespace M::motr::Flags::detail {

ALWAYS_INLINE inline motr::Time::Duration
spinWaitOnFlagTimeout(std::atomic<uint64_t> &timeout) {
  using Duration = motr::Time::Duration;
  using Timestamp = motr::Time::Timestamp;
  using Elapsed = motr::Time::Elapsed;

  // MOTR_LOG("timeout: {:p} {:p}", (void*)(&value), (void*)(&timeout));
  const Duration maxWait = Duration::fromSeconds(1000);
  const Duration cycleTime = Duration::fromSeconds(0.01);

  Elapsed elapsedTimer;
  Timestamp flagTimeout;
  Duration duration;
  Duration elapsed;

  while (true) {
    flagTimeout.v = timeout.load(std::memory_order_relaxed);

    if (flagTimeout.v == 0)
      return elapsed;

    elapsed = elapsedTimer.elapsed();

    if (elapsed > maxWait)
      break;

    duration.v = flagTimeout.v;
    // if the timeout is less than 1 hour, then assume it is a relative timeout
    if (duration < Duration::fromSeconds(60 * 60)) {
      // and convert it to an absolute timeout
      flagTimeout = elapsedTimer.t1 + duration;
      timeout.store(flagTimeout.v, std::memory_order_relaxed);
    }

    // now duration is the time remaining until the timeout
    duration = flagTimeout - elapsedTimer.t1;

    if (duration.v <= 0)
      break;

    if (duration < cycleTime)
      duration.sleep();
    else
      cycleTime.sleep();
  }
  timeout.store(0, std::memory_order_relaxed);
  return elapsed;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
template <typename ValueType>
ALWAYS_INLINE ValueType getFlagValue(const std::atomic<ValueType> &value) {
  if constexpr (std::is_same<ValueType, bool>::value) {
    return value.load(std::memory_order_relaxed) != 0;
  } else {
    return value.load(std::memory_order_relaxed);
  }
}

// returns the previous value
template <typename ValueType>
ALWAYS_INLINE ValueType setFlagValue(std::atomic<ValueType> &value,
                                     const ValueType &v) {
#pragma GCC diagnostic pop
  // todo: add flag bank generation id increment
  std::atomic<ValueType> newValue;
  if constexpr (std::is_same<ValueType, bool>::value) {
    newValue.store(v ? 1 : 0, std::memory_order_relaxed);
  } else {
    newValue.store(v, std::memory_order_relaxed);
  }

  std::atomic<ValueType> result = value.exchange(
      newValue.load(std::memory_order_relaxed), std::memory_order_relaxed);

  if constexpr (std::is_same<ValueType, bool>::value) {
    return result != 0;
  } else {
    return result;
  }
}
template <typename ValueType>
struct AtomicFlag {
  using UnderlyingType = ValueType;
  std::atomic<UnderlyingType> value;

  ALWAYS_INLINE ValueType get() const {
    return detail::getFlagValue<ValueType>(value);
  }

  ALWAYS_INLINE ValueType
  getAfterWaitingOn(AtomicFlag<ValueType> &timeout) const {
    auto waitTime = detail::spinWaitOnFlagTimeout(timeout.value);
    if (waitTime.v > 0) {
      MOTR_LOG("WARNING: flag timeout waited {} / {}", waitTime.toString(),
               waitTime.v);
    }
    return get();
  }

  ALWAYS_INLINE ValueType set(const ValueType &v) {
    return detail::setFlagValue<ValueType>(value, v);
  }

  ALWAYS_INLINE ValueType operator=(const ValueType &v) { return set(v); }

  ALWAYS_INLINE operator ValueType() const { return get(); }

  static AtomicFlag<ValueType> &getInvalidPlaceholder(int col) {
    // this much match Manager::cols
    static constexpr size_t cols = 4;
    static AtomicFlag<ValueType> placeholder[cols] = {
        {detail::UNINITIALIZED_ATOMIC_FLAG_VALUE},
        {detail::UNINITIALIZED_ATOMIC_FLAG_VALUE},
        {detail::UNINITIALIZED_ATOMIC_FLAG_VALUE},
        {detail::UNINITIALIZED_ATOMIC_FLAG_VALUE}};

    return placeholder[col % cols];
  }

  ALWAYS_INLINE static bool
  isInvalidPlaceholder(const AtomicFlag<ValueType> &value) {
    // todo: change this if valueCol
    return &value == &getInvalidPlaceholder(1);
  }

  // note: the flag can be valid but not initialized
  // so check initialized() to make sure the flag is fully initialized
  [[nodiscard]] ALWAYS_INLINE bool valid() const {
    return !isInvalidPlaceholder(*this);
  }

  [[nodiscard]] ALWAYS_INLINE bool initialized() const {
    // note: the flag can be initialized but not valid
    // so check valid() to make sure the flag is fully initialized
    return value.load(std::memory_order_relaxed) !=
           detail::UNINITIALIZED_ATOMIC_FLAG_VALUE;
  }

  // return true if the flag was already to the uninitialized value
  ALWAYS_INLINE bool setUninitialized() {
    return set(detail::UNINITIALIZED_ATOMIC_FLAG_VALUE) ==
           detail::UNINITIALIZED_ATOMIC_FLAG_VALUE;
  }
};

} // namespace M::motr::Flags::detail

struct M::motr::Flags::Manager {
  using UnderlyingType = uint64_t;
  using AtomicType = detail::AtomicFlag<UnderlyingType>;
  using SharedMemory = TypedSharedMemory<AtomicType>;
  static_assert(sizeof(AtomicType) == sizeof(UnderlyingType),
                "AtomicType must be the same size as UnderlyingType");
  static constexpr size_t rows = 1024;
  static constexpr size_t cols = 4;
  static constexpr size_t hashCol = 0;
  static constexpr size_t valueCol = 1;
  static constexpr size_t getTimeoutCol = 2;
  static constexpr size_t countCol = 3;
  static constexpr size_t capacity = rows * cols;
  static constexpr size_t numBytes = capacity * sizeof(AtomicType);

  ALWAYS_INLINE static SharedMemoryInit getSHMInitMode(const char *name,
                                                       size_t len) {
    SharedMemory tmp(SharedMemoryInit::OpenExisting, name, len);
    if (tmp.valid())
      return SharedMemoryInit::OpenExisting;
    return SharedMemoryInit::ExclusiveCreate;
  }

  ALWAYS_INLINE static SharedMemory &getSharedMemory() {
    static SharedMemoryInit init =
        getSHMInitMode(detail::getSharedMemoryName().c_str(), numBytes);
    static SharedMemory sharedMemory(init, detail::getSharedMemoryName(),
                                     numBytes);
    return sharedMemory;
  }

  ALWAYS_INLINE static void resetSharedMemory() {
    auto &sharedMemory = getSharedMemory();
    sharedMemory.cleanup();
    // Create and destroy a new shared memory region
    // to force cleanup of any existing regions
    SharedMemory(SharedMemoryInit::ExclusiveCreate,
                 detail::getSharedMemoryName(), numBytes);
  }

  template <uint64_t HASH>
  ALWAYS_INLINE static uint64_t findRowOfHash() {
    auto &sharedMemory = getSharedMemory();
    constexpr const uint64_t begRow = (HASH % rows);
    constexpr const uint64_t endRow = (HASH % rows) + rows;
    for (uint64_t row = begRow; row < endRow; ++row) {
      auto &curHash = sharedMemory[(row * cols + hashCol) % capacity];
      if (curHash == HASH) {
        return row;
      }
    }
    return -1;
  }

  // linear probe of hash value
  // returns -1 if not found after all rows are checked
  //
  // if createIfMissing is true, then the hash will be set in the first empty
  // row found
  // and the function will return the row index of the empty row
  //
  // if no empty rows are left after searching the entire array
  // then the function will return -1

  // if createIfMissing is false, then the hash will not be set
  // and the function will return -1 if the hash is not found
  ALWAYS_INLINE static uint64_t findOrCreateRowOfHash(uint64_t hash,
                                                      bool createIfMissing) {
    auto &sharedMemory = getSharedMemory();
    uint64_t row = (hash % rows);
    const uint64_t endRow = row + rows;
    while (row < endRow) {
      auto &curHash = sharedMemory[(row * cols) % capacity + hashCol];
      if (curHash == hash) {
        return row;
      }
      // if the current row is empty, set the hash and return the row
      if (createIfMissing) {
        uint64_t expected = 0x0;
        if (curHash.value.compare_exchange_strong(expected, hash)) {
          return row;
        }
      }
      ++row;
    }
    return -1;
  }

  template <uint64_t HASH>
  ALWAYS_INLINE static uint64_t getRow() {
    static const uint64_t row = findRowOfHash<HASH>();
    return row;
  }

  // non-compile time version
  ALWAYS_INLINE static AtomicType &getAtomicRef(uint64_t hash, int col,
                                                bool createIfMissing) {
    const uint64_t row = findOrCreateRowOfHash(hash, createIfMissing);
    col = col % cols;
    if (row == uint64_t(-1)) {
      // no row found, (or space ran out on createIfMissing = true)
      // return a placeholder
      // CAREFUL! this is shared across all invalid hashes!
      AtomicType &placeholder = AtomicType::getInvalidPlaceholder(col);
      // always uninitialize the placeholder
      if (placeholder.set(detail::UNINITIALIZED_ATOMIC_FLAG_VALUE) !=
          detail::UNINITIALIZED_ATOMIC_FLAG_VALUE) {
        // if the placeholder was already uninitialized, then we need to
        // then somewhere someone got an invalid flag and still proceeded
        // to use it
        MOTR_LOG("ERROR: placeholder was already uninitialized", "");
      }

      return placeholder;
    }
    return getSharedMemory()[row * cols + col];
  }
};

template <const char *NAME, typename T>
// template <char const* NAME, typename T,
// std::enable_if_t<std::is_unsigned<T>::value || std::is_same<T,
// bool>::value, int>::value>
struct M::motr::Flags::FlagT<NAME, T,
                             std::enable_if_t<std::is_unsigned<T>::value ||
                                              std::is_same<T, bool>::value>> {

  using UnderlyingType = T;
  using AtomicType = detail::AtomicFlag<UnderlyingType>;

  static constexpr const char *name() { return NAME; }
  static constexpr uint64_t hash() {
    constexpr Hash::Value nameHash{NAME};
    return nameHash.v;
  }

  AtomicType &value;
  AtomicType &getTimeout;
  AtomicType &count;

  FlagT()
      : value(Manager::getAtomicRef(hash(), Manager::valueCol, true)),
        getTimeout(Manager::getAtomicRef(hash(), Manager::getTimeoutCol, true)),
        count(Manager::getAtomicRef(hash(), Manager::countCol, true)) {}

  // create a flags for a possibly invalid flag
  FlagT(decltype(nullptr))
      : value(Manager::getAtomicRef(hash(), Manager::valueCol, false)),
        getTimeout(
            Manager::getAtomicRef(hash(), Manager::getTimeoutCol, false)),
        count(Manager::getAtomicRef(hash(), Manager::countCol, false)) {}

  // No copy, no move
  FlagT(const FlagT &) = delete;
  FlagT &operator=(const FlagT &) = delete;
  FlagT(FlagT &&) = delete;
  FlagT &operator=(FlagT &&) = delete;

  // Getters
  ALWAYS_INLINE UnderlyingType getNoWait() const { return value.get(); }
  ALWAYS_INLINE UnderlyingType get() const {
    return value.getAfterWaitingOn(getTimeout);
  }
  ALWAYS_INLINE operator UnderlyingType() const {
    return value.getAfterWaitingOn(getTimeout);
  }

  ALWAYS_INLINE UnderlyingType set(const UnderlyingType &v) {
    return value.set(v);
  }
  ALWAYS_INLINE auto operator=(const UnderlyingType &v) { return value.set(v); }

  // getTimeout
  ALWAYS_INLINE int64_t setGetTimeout(int64_t timeout) {
    return getTimeout.set(timeout);
  }
  ALWAYS_INLINE int64_t getGetTimeout() const { return getTimeout.get(); }

  ALWAYS_INLINE bool valid() const { return value.valid(); }
  ALWAYS_INLINE bool initialized() const { return value.initialized(); }
};

struct M::motr::Flags::Flag {

  using UnderlyingType = uint64_t;
  using AtomicType = detail::AtomicFlag<UnderlyingType>;

  uint64_t nameHash;
  std::string nameStr;
  AtomicType &value;
  AtomicType &getTimeout;
  AtomicType &count;

  Flag(std::string_view name)
      : nameHash(Hash::Value{name}.v), nameStr(name),
        value(Manager::getAtomicRef(nameHash, Manager::valueCol, false)),
        getTimeout(
            Manager::getAtomicRef(nameHash, Manager::getTimeoutCol, false)),
        count(Manager::getAtomicRef(nameHash, Manager::countCol, false)) {}

  Flag(std::string_view name, uint64_t defaultValue)
      : nameHash(Hash::Value{name}.v), nameStr(name),
        value(Manager::getAtomicRef(nameHash, Manager::valueCol, true)),
        getTimeout(
            Manager::getAtomicRef(nameHash, Manager::getTimeoutCol, true)),
        count(Manager::getAtomicRef(nameHash, Manager::countCol, true)) {
    if (valid()) {
      UnderlyingType expected = detail::UNINITIALIZED_ATOMIC_FLAG_VALUE;
      value.value.compare_exchange_strong(expected, defaultValue);
    }
  }

  ALWAYS_INLINE uint64_t hash() { return nameHash; }
  ALWAYS_INLINE const char *name() { return nameStr.c_str(); }

  // Get Value
  ALWAYS_INLINE UnderlyingType getNoWait() const { return value.get(); }
  ALWAYS_INLINE UnderlyingType get() const {
    return value.getAfterWaitingOn(getTimeout);
  }
  ALWAYS_INLINE operator UnderlyingType() const {
    return value.getAfterWaitingOn(getTimeout);
  }

  // Set Value
  ALWAYS_INLINE UnderlyingType set(const UnderlyingType &v) {
    return value.set(v);
  }
  ALWAYS_INLINE Flag &operator=(const UnderlyingType &v) {
    value.set(v);
    return *this;
  }

  // getTimeout
  ALWAYS_INLINE int64_t setGetTimeout(int64_t timeout) {
    return getTimeout.set(timeout);
  }
  ALWAYS_INLINE int64_t getGetTimeout() const { return getTimeout.get(); }

  ALWAYS_INLINE bool valid() const { return value.valid(); }
  ALWAYS_INLINE bool initialized() const { return value.initialized(); }
};

#endif
