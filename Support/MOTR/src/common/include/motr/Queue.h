//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_QUEUE_H
#define MOTR_QUEUE_H

#include "motr/Log.h"
#include "motr/Macros.h"
#include "motr/Message.h"
#include "motr/SharedMemory.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <typeinfo>
#include <unistd.h>
#include <vector>

// static_assert(std::atomic<uint64_t>::is_lock_free,
//               "Queue must have lock free atomic types");
namespace M::motr {
template <typename T>
struct Queue;
struct StringQueue;
struct StringQueueResult;
namespace detail {
struct QueueControl;

template <typename T>
struct QueueMemory;
} // namespace detail
}; // namespace M::motr

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "Atomic uint64_t is not always lock free");

struct M::motr::detail::QueueControl {
  // Queue Design Overview:
  // The Queue is designed as a thread-safe data structure that allows multiple
  // producers to add elements and a single consumer to retrieve them. It
  // operates in a circular buffer manner, ensuring efficient memory usage and
  // minimizing contention between threads.
  //
  // Key Features:
  // - Lock-free atomic operations for state management, ensuring high
  // performance.
  // - The queue maintains a read head and a write head to track the positions
  // for reading
  //   and writing elements, respectively.
  // - It supports operations to check the number of elements available to read
  // and write,
  //   as well as methods to initialize the queue and print debug information.
  //
  // Existing Behavior:
  // - When the queue is full (i.e., the number of elements written equals the
  // capacity),
  //   the `send` method will return an error or zero, indicating that no more
  //   elements can be added until space is available. This behavior is
  //   currently non-blocking.
  //
  // - If the queue exhausts its underlying storage (e.g., due to a memory
  // allocation failure),
  //   the `send` method logs an error and returns an indication of failure,
  //   ensuring that the system can recover from such situations without
  //   crashing.
  //
  // - When the `recv` method is called and the queue is empty, it will return
  // an empty result,
  //   indicating that there are no elements available to read. This behavior
  //   prevents the consumer from entering an invalid state or crashing due to
  //   attempting to read non-existent data.
  //
  // Pathological Use Cases:
  // - If a producer continuously sends data without allowing the consumer to
  // read,
  //   the queue may become full, leading to potential deadlocks or resource
  //   exhaustion. Implementing backpressure mechanisms can help mitigate this
  //   issue by signaling producers to slow down or pause until the consumer
  //   catches up.
  //
  // - If a consumer attempts to read from an empty queue, it will return an
  // empty result,
  //   ensuring that the consumer does not enter an invalid state or crash.
  //
  // This design allows for efficient communication between threads in a
  // concurrent environment.
  enum States : uint64_t {
    UNINITIALIZED = 0,
    INITIALIZING = 1,
    INITIALIZED = 2,
  };

  std::atomic<uint64_t> state;     // Current state of the queue (UNINITIALIZED,
                                   // INITIALIZING, INITIALIZED).
  std::atomic<uint64_t> readHead;  // Index of the next element to be read
  std::atomic<uint64_t> writeHead; // Index of the next position to write
  std::atomic<uint64_t> numConsumed;  // Count of elements consumed
  std::atomic<uint64_t> numPublished; // Count of elements published

  // Returns the number of elements available to read from the queue.
  size_t availableToRead() const;

  // Returns the number of elements currently taken from the queue.
  size_t taken() const;

  // Prints debug information about the queue, including its size.
  void debugPrint(size_t size) const;

  // Initializes the queue control structure and sets the state to INITIALIZING.
  // Returns true if initialization is successful, false otherwise.
  bool init();
};

inline size_t M::motr::detail::QueueControl::availableToRead() const {
  const auto published = numPublished.load(std::memory_order_relaxed);
  const auto consumed = numConsumed.load(std::memory_order_relaxed);
  assert(published >= consumed);
  return published - consumed;
}

inline size_t M::motr::detail::QueueControl::taken() const {
  const auto writehead = writeHead.load(std::memory_order_relaxed);
  const auto consumed = numConsumed.load(std::memory_order_relaxed);
  assert(writehead >= consumed);
  return writehead - consumed;
}

template <typename T>
struct M::motr::detail::QueueMemory {
  QueueMemory(SharedMemoryInit mode, const std::string &name, size_t capacity);

  // disallow copy, allow move
  QueueMemory(const QueueMemory &) = delete;
  QueueMemory &operator=(const QueueMemory &) = delete;
  QueueMemory(QueueMemory &&other) = default;
  QueueMemory &operator=(QueueMemory &&other) = default;

  bool valid() const;

  std::string name;
  TypedSharedMemory<detail::QueueControl> controlMemory;
  detail::QueueControl &control;
  TypedSharedMemory<T> bufferMemory;
  T *buffer;
};

template <typename T>
M::motr::detail::QueueMemory<T>::QueueMemory(SharedMemoryInit mode,
                                             const std::string &name,
                                             size_t capacity)
    : name(name),                                   //
      controlMemory(mode, name + "_ctrl", 1),       //
      control(controlMemory[0]),                    //
      bufferMemory(mode, name + "_data", capacity), //
      buffer(&bufferMemory[0]) {}                   //

template <typename T>
bool M::motr::detail::QueueMemory<T>::valid() const {
  return controlMemory.valid() && bufferMemory.valid();
}

// M::Queue<T>
//
// Threadsafe queue
// Multiple producers, single consumer
// (may be multiple consumer safe, but not guaranteed)
template <typename T>
struct M::motr::Queue : public detail::QueueMemory<T> {
  using Base = detail::QueueMemory<T>;

  Queue(SharedMemoryInit mode, const std::string &name, size_t capacity);
  ~Queue() = default;

  size_t send(const T *first, size_t count, bool publish = true);
  size_t send(const std::vector<std::string_view> &strings);
  size_t send(const T &message);

  std::vector<T> recv(size_t maxcount);

  // these methods are thread safe but not guaranteed to be
  // correct upon return (e.g. another thread could have changed the state)
  size_t numAvailableToRead() const;
  size_t numAvailableToWrite() const;

  size_t capacity() const;

  bool empty() const;
  bool full() const;
  bool valid() const;
  void debugPrint() const;

  detail::QueueControl &control;
};

struct M::motr::StringQueueResult {
  using StringHeaders = std::vector<const StringHeader *>;
  using StringViews = std::vector<std::string_view>;

  StringQueueResult() = default;
  StringQueueResult(StringHeaders &&headers, StringViews &&views,
                    size_t totalBytes, detail::QueueControl *control)
      : headers(std::move(headers)), views(std::move(views)),
        totalBytes(totalBytes), control(control) {}
  StringQueueResult(const StringQueueResult &) = delete;
  StringQueueResult &operator=(const StringQueueResult &) = delete;
  StringQueueResult(StringQueueResult &&other) noexcept = delete;
  StringQueueResult &operator=(StringQueueResult &&other) noexcept = delete;
  ~StringQueueResult();

  bool valid() const { return control != nullptr; }

  void debugPrint() const {
    size_t N = headers.size();
    assert(N == views.size());
    for (size_t i = 0; i < N; i++) {
      auto &header = headers[i];
      auto &view = views[i];
      MOTR_LOG("recv[{}]: ##{:016x} \"{}\"", i, header->hashId, view);
    }
  }

  std::pair<uint64_t, std::string_view> operator[](size_t index) const {
    return {headers[index]->hashId, views[index]};
  }

  StringHeaders headers = {};
  StringViews views = {};
  size_t totalBytes = 0;
  detail::QueueControl *control = nullptr;
};

struct M::motr::StringQueue : public detail::QueueMemory<char> {
  using Base = detail::QueueMemory<char>;
  using StringViews = std::vector<std::string_view>;

  template <typename T>
  StringQueue(Queue<T> &src)
      : Base(SharedMemoryInit::OpenExisting, src.name, src.bufferMemory.size),
        control(Base::control)

  {
    assert(src.valid() == valid());
  }

  ~StringQueue() = default;

  size_t send(const StringViews &);
  StringQueueResult recv();

  void debugPrint() const;

  detail::QueueControl &control;
};

inline bool M::motr::detail::QueueControl::init() {
  uint64_t expected = States::UNINITIALIZED;
  if (!state.compare_exchange_strong(expected, States::INITIALIZING)) {
    MOTR_LOG("QueueControl::init() failed: state is not UNINITIALIZED", "");
    return false;
  }

  readHead.store(0, std::memory_order_relaxed);
  writeHead.store(0, std::memory_order_relaxed);
  numConsumed.store(0, std::memory_order_relaxed);
  numPublished.store(0, std::memory_order_relaxed);

  expected = States::INITIALIZING;
  if (!state.compare_exchange_strong(expected, States::INITIALIZED)) {
    MOTR_LOG("QueueControl::init() failed: state is not INITIALIZING", "");
    return false;
  }

  return true;
}

template <typename T>
M::motr::Queue<T>::Queue(SharedMemoryInit mode, const std::string &name,
                         size_t capacity)
    : Base(mode, name, capacity), control(Base::control) {
  if (!valid())
    return;

  bool owner = mode == SharedMemoryInit::ExclusiveCreate;
  if (owner) {
    if (!control.init()) {
      MOTR_LOG("Queue<>[{}] failed to initialize control region", name);
      return;
    }
  }
}

template <typename T>
MOTR_ALWAYS_INLINE size_t M::motr::Queue<T>::capacity() const {
  return Base::bufferMemory.capacity();
}

template <typename T>
bool M::motr::Queue<T>::valid() const {
  return Base::valid() && capacity() > 0;
}

template <typename T>
bool M::motr::Queue<T>::empty() const {
  return control.readHead.load(std::memory_order_relaxed) >=
         control.writeHead.load(std::memory_order_relaxed);
}

template <typename T>
bool M::motr::Queue<T>::full() const {
  return control.writeHead.load(std::memory_order_relaxed) -
             control.readHead.load(std::memory_order_relaxed) >=
         capacity();
}

template <typename T>
size_t M::motr::Queue<T>::numAvailableToRead() const {

  const auto published = control.numPublished.load(std::memory_order_relaxed);
  const auto consumed = control.numConsumed.load(std::memory_order_relaxed);
  assert(published >= consumed);
  return published - consumed;
}

template <typename T>
size_t M::motr::Queue<T>::numAvailableToWrite() const {
  const auto writeHead = control.writeHead.load(std::memory_order_relaxed);
  const auto numConsumed = control.numConsumed.load(std::memory_order_relaxed);

  // Handle wrap-around using modulo arithmetic
  const auto capacity =
      this->capacity(); // Assuming capacity() returns the size of the buffer
  const auto available = (writeHead >= numConsumed)
                             ? (capacity - (writeHead - numConsumed))
                             : (numConsumed - writeHead);
  return available;
}

inline void M::motr::detail::QueueControl::debugPrint(size_t size) const {
  const auto _writeHead = writeHead.load(std::memory_order_relaxed);
  const auto _numPublished = numPublished.load(std::memory_order_relaxed);
  const auto _readHead = readHead.load(std::memory_order_relaxed);
  const auto _numConsumed = numConsumed.load(std::memory_order_relaxed);

  const auto partialWrites = _writeHead - _numPublished;
  const auto partialReads = _readHead - _numConsumed;
  const auto taken = _writeHead - _numConsumed;
  const auto remaining = size ? size - taken : 0;

  const auto s = size ? size : 1;

  MOTR_LOG("size={}", s);
  MOTR_LOG("writeHead={} ({})", _writeHead, _writeHead % s);
  MOTR_LOG("numPublished={} ({})", _numPublished, _numPublished % s);
  MOTR_LOG("readHead={} ({})", _readHead, _readHead % s);
  MOTR_LOG("numConsumed={} ({})", _numConsumed, _numConsumed % s);
  MOTR_LOG("partialWrites={}", partialWrites);
  MOTR_LOG("partialReads={}", partialReads);
  MOTR_LOG("taken={}", taken);
  MOTR_LOG("remaining={}", remaining);
}

template <typename T>
inline void M::motr::Queue<T>::debugPrint() const {
  MOTR_LOG("queue.shm.control.name={}", Base::controlMemory.name);
  MOTR_LOG("queue.shm.control.size={}", Base::controlMemory.size);
  MOTR_LOG("queue.shm.control.stride={}", Base::controlMemory.stride);
  MOTR_LOG("queue.shm.control.fd={}", Base::controlMemory.fd);
  MOTR_LOG("queue.shm.control.mode={}", int(Base::controlMemory.mode));
  MOTR_LOG("queue.shm.control.data={}", fmt::ptr(Base::controlMemory.data));

  MOTR_LOG("queue.shm.buffer.name={}", Base::bufferMemory.name);
  MOTR_LOG("queue.shm.buffer.size={}", Base::bufferMemory.size);
  MOTR_LOG("queue.shm.buffer.stride={}", Base::bufferMemory.stride);
  MOTR_LOG("queue.shm.buffer.fd={}", Base::bufferMemory.fd);
  MOTR_LOG("queue.shm.buffer.mode={}", int(Base::bufferMemory.mode));
  MOTR_LOG("queue.shm.buffer.data={}", fmt::ptr(Base::bufferMemory.data));
  control.debugPrint(capacity());
}

inline void M::motr::StringQueue::debugPrint() const {
  MOTR_LOG("queue.shm.control.name={}", controlMemory.name);
  MOTR_LOG("queue.shm.control.size={}", controlMemory.size);
  MOTR_LOG("queue.shm.control.stride={}", controlMemory.stride);
  MOTR_LOG("queue.shm.control.fd={}", controlMemory.fd);
  MOTR_LOG("queue.shm.control.mode={}", int(controlMemory.mode));
  MOTR_LOG("queue.shm.control.data={}", fmt::ptr(controlMemory.data));

  MOTR_LOG("queue.shm.buffer.name={}", bufferMemory.name);
  MOTR_LOG("queue.shm.buffer.size={}", bufferMemory.size);
  MOTR_LOG("queue.shm.buffer.stride={}", bufferMemory.stride);
  MOTR_LOG("queue.shm.buffer.fd={}", bufferMemory.fd);
  MOTR_LOG("queue.shm.buffer.mode={}", int(bufferMemory.mode));
  MOTR_LOG("queue.shm.buffer.data={}", fmt::ptr(bufferMemory.data));
  control.debugPrint(bufferMemory.capacity());
}

// TODO: hoist this to a common header
inline size_t align(size_t size, size_t alignment = 16) {
  return (size + (alignment - 1)) &
         ~(alignment - 1); // Align to specified alignment
}

inline size_t M::motr::StringQueue::send(const StringViews &string_views) {
  // Check if the StringQueue is valid before proceeding
  if (!valid())
    return 0;

  // Calculate the total size of the strings, aligning each size to 16 bytes
  size_t totalSize = 0; // Total size of all strings to be sent
  for (auto &sv : string_views) {
    size_t aligned_size = align(sv.size());
    totalSize += aligned_size;
  }
  assert(totalSize % 16 == 0); // Ensure the total size is a multiple of 16

  const size_t size = Base::bufferMemory.capacity(); // Get the buffer capacity

  // Calculate available space in the buffer
  [[maybe_unused]] const auto available = size - control.taken();
  assert(totalSize <= available); // Ensure we have enough space
  assert(totalSize <= size);      // Ensure we do not exceed buffer capacity

  // TODO: Handle partial sends if totalSize exceeds available space

  // Reserve slots for this thread by atomically updating writeHead
  uint64_t writeBeg =
      control.writeHead.fetch_add(totalSize, std::memory_order_acq_rel);
  uint64_t writePos = writeBeg;             // Current write position
  uint64_t writeEnd = writePos + totalSize; // End position for writing

  // Check for race conditions where another producer may have published more
  // data
  assert(writePos - control.numConsumed.load(std::memory_order_relaxed) <=
         size);

  size_t sv_index = 0; // Index for the current string view being processed
  while (writePos < writeEnd) {
    uint64_t writeIndex =
        writePos % size; // Calculate the index in the circular buffer
    char *dst = buffer + writeIndex;   // Destination pointer for writing
    auto &sv = string_views[sv_index]; // Current string view
    size_t sv_size = sv.size();        // Size of the current string view

    // Check if the string fits in the remaining space in the buffer
    uint64_t remaining = size - writeIndex;
    if (sv_size <= remaining) {
      std::copy(sv.data(), sv.data() + sv_size, dst);
      dst += sv_size;
    } else {
      // Handle wrap-around case (not implemented yet)
      assert(false && "TODO: sending string view across ring buffer wrap "
                      "boundary not implemented");
      std::copy(sv.data(), sv.data() + remaining, dst);
      dst = &buffer[0];
      std::copy(sv.data() + remaining, sv.data() + sv_size, dst);
      dst += sv_size - remaining;
    }

    // Align the size to the next multiple of 16
    sv_size = align(sv_size);
    writePos += sv_size;
    assert(writePos % 16 == 0);

    sv_index++; // Move to the next string view

    // Calculate how much has been written
    [[maybe_unused]] uint64_t written = writePos - writeBeg;
    assert(written <= totalSize);
    assert(sv_index <= string_views.size());
  }
  assert(writePos == writeEnd); // Ensure we wrote the expected amount

  // NOTE: Ensure that numPublished matches writePos before proceeding. This
  // guarantees that the consumer is aware of all newly published data,
  // preventing it from reading stale or incomplete information.  Such
  // inconsistencies can lead to undefined behavior in a multi-threaded
  // environment. While multiple threads can concurrently push data into the
  // queue by atomically reserving their segments, consumers must adhere to the
  // submission order to avoid accessing data that is not yet ready for
  // consumption.
  while (control.numPublished.load(std::memory_order_acquire) != writeBeg)
    std::this_thread::yield();
  control.numPublished.fetch_add(totalSize, std::memory_order_release);

  return totalSize;
}

template <typename T>
void check_alignment(const T *ptr) {
  assert((reinterpret_cast<uintptr_t>(ptr) & 15) == 0 &&
         "Pointer is not 16-byte aligned");
}

inline M::motr::StringQueueResult M::motr::StringQueue::recv() {
  // Notes:
  //   To maintain 16 byte alignment for StringHeader, all following strings
  //   are padded to 16 bytes as well.
  //   Strings cannot by split by the buffer wrap boundary.
  //   because the return value is a vector of string views,
  if (!valid())
    return {};

  // TODO(rparolin): Move this to the StringHeader class for proper
  // encapsulation.
  constexpr const size_t stride = sizeof(StringHeader);
  static_assert(stride == 16, "StringHeader stride is not 16");

  size_t numBytesAvailable = control.availableToRead();
  assert(numBytesAvailable % stride == 0);
  if (numBytesAvailable <= 0)
    return {};

  StringViews views;
  StringQueueResult::StringHeaders headers;

  const uint64_t readBeg =
      control.readHead.fetch_add(numBytesAvailable, std::memory_order_acq_rel);
  const uint64_t readEnd = readBeg + numBytesAvailable;
  uint64_t readPos = readBeg;

  size_t size = Base::bufferMemory.capacity();
  [[maybe_unused]] size_t totalStringBytes = 0;
  [[maybe_unused]] size_t totalPadding = 0;
  size_t totalBytes = 0;

  check_alignment(buffer);
  while (readPos < readEnd) {
    uint64_t readIndex = readPos % size;
    assert(readIndex + stride <= size);
    assert(readIndex % 16 == 0);

    const char *readPtr = buffer + readIndex;
    check_alignment(readPtr);

    // refer to definition of StringHeader in Message.h
    const StringHeader *headerPtr =
        reinterpret_cast<const StringHeader *>(readPtr);
    uint32_t svSize = headerPtr->size;

    assert(headerPtr->header == StringHeader::Header);
    assert(svSize > 0);
    assert(headerPtr->hashId != 0);
    [[maybe_unused]] size_t idx = views.size();

    // push back string view of the StringHeader msg
    // views.emplace_back(readPtr, stride);
    headers.emplace_back(headerPtr);

    totalBytes += stride;
    readPos += stride;
    readIndex = readPos % size;
    assert(readIndex % 16 == 0);

    readPtr = buffer + readIndex;
    check_alignment(readPtr);
    assert(readIndex + svSize <= size);

    views.emplace_back(readPtr, svSize);

    size_t padding =
        (stride - (svSize % stride)) % stride; // Correct padding calculation
    readPos += svSize + padding;
    totalPadding += padding;
    totalStringBytes += svSize;
    totalBytes += svSize + padding;
  }
  assert(readPos == readEnd);
  assert(totalBytes == numBytesAvailable);
  assert(totalStringBytes + totalPadding + stride * headers.size() ==
         numBytesAvailable);

  return {std::move(headers), std::move(views), totalBytes, &control};
}

inline M::motr::StringQueueResult::~StringQueueResult() {
  if (control != nullptr && totalBytes > 0)
    control->numConsumed.fetch_add(totalBytes, std::memory_order_release);
}

template <typename T>
size_t M::motr::Queue<T>::send(const T *ptr, size_t count, bool publish) {
  if (!valid())
    return 0;

  if (!ptr)
    return 0; // noop if user sends a nullptr

  const size_t size = capacity();

  // Atomically check and reserve space
  while (true) {
    uint64_t writeHead = control.writeHead.load(std::memory_order_relaxed);
    uint64_t numConsumed = control.numConsumed.load(std::memory_order_acquire);
    size_t available = (writeHead >= numConsumed)
                           ? (size - (writeHead - numConsumed))
                           : (numConsumed - writeHead);

    if (count > available || count > size)
      return 0; // Not enough space

    // Try to reserve space atomically
    if (control.writeHead.compare_exchange_weak(writeHead, writeHead + count,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed)) {
      // Reservation succeeded, writeHead is the starting index
      uint64_t writeIndex = writeHead % size;
      auto &buffer = Base::buffer;
      uint64_t remaining = size - writeIndex;

      if (count <= remaining) {
        std::copy(ptr, ptr + count, &buffer[writeIndex]);
      } else {
        std::copy(ptr, ptr + remaining, &buffer[writeIndex]);
        std::copy(ptr + remaining, ptr + count, &buffer[0]);
      }

      if (publish) {
        while (control.numPublished.load(std::memory_order_acquire) !=
               writeHead)
          std::this_thread::yield();
        control.numPublished.fetch_add(count, std::memory_order_release);
      }
      return count;
    }
    // If reservation failed, retry
    std::this_thread::yield();
  }
}

template <typename T>
size_t M::motr::Queue<T>::send(const T &message) {
  return send(&message, 1);
}

// NOTE: DO not use this method, it is not thread safe yet
template <typename T>
size_t
M::motr::Queue<T>::send(const std::vector<std::string_view> &string_views) {
  size_t total_size = 0;
  for (auto &sv : string_views) {
    total_size += sv.size();
    send(sv.data(), sv.size(),
         false); // defer publish until all strings are sent
  }
  send(nullptr, total_size, true); // publish all strings
  return total_size;
}

template <typename T>
std::vector<T> M::motr::Queue<T>::recv(size_t maxcount) {
  // single consumer means some calculations will not change as we read
  // so we can safely preallocate the result vector
  // available = published - consumed
  // todo: Unless we need MPMC, we need to add extra checks to ensure
  // this queue is only ever consumed by a single thread

  // How many elements are ready to be consumed
  size_t count = control.availableToRead();

  // no data available
  if (count == 0)
    return {};

  // if maxcount == 0, read all available data
  if (maxcount != 0 && maxcount < count)
    // otherwise, read at most maxcount elements
    count = maxcount;

  assert(count <= maxcount);

  std::vector<T> result(count);
  const uint64_t readPos =
      control.readHead.fetch_add(count, std::memory_order_acq_rel);
  const size_t size = capacity();

  // wrap the calculation of read index into ringbuffer index space
  const uint64_t readIndex = readPos % size;

  // Calculate how many elements until ring buffer end are available
  const uint64_t chunksize = size - readIndex;

  auto &buffer = Base::buffer;

  if (chunksize >= count)
    std::copy(&buffer[readIndex], &buffer[readIndex + count], result.begin());
  else {
    // must copy the first chunk and the second chunk across the ring buffer
    // wrap copy the first chunk
    std::copy(&buffer[readIndex], &buffer[readIndex + chunksize],
              result.begin());
    // copy the second chunk
    std::copy(&buffer[0], &buffer[(count - chunksize)], &result[chunksize]);
  }

  // update the number of consumed elements to
  // indicate to the producer that the data has been consumed
  // freeing it to be written over by producers
  control.numConsumed.fetch_add(count, std::memory_order_release);
  return result;
}

#endif
