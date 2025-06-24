//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_SHARED_MEMORY_H
#define MOTR_SHARED_MEMORY_H

#include "motr/Common.h"
#include "motr/Log.h"

#ifdef MOTR_PLATFORM_MACOS
#include <sys/posix_shm.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>

#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <set>
#include <string>

#if !defined(PSHMNAMLEN)
#error PSHMNAMLEN is not defined
#endif

namespace M {

namespace detail {
inline std::set<std::string> &getShmRegistry() {
  static std::set<std::string> shm_registry;
  return shm_registry;
}

inline std::mutex &getShmRegistryMutex() {
  static std::mutex shm_registry_mutex;
  return shm_registry_mutex;
}

inline void shm_cleanup_all() {
  std::lock_guard<std::mutex> lock(getShmRegistryMutex());
  for (const auto &name : getShmRegistry()) {
    shm_unlink(name.c_str());
  }
  getShmRegistry().clear();
}

inline void shm_signal_handler(int) {
  shm_cleanup_all();
  std::_Exit(128); // exit immediately, don't run other handlers
}

inline void shm_register_signal_handlers() {
  static bool registered = false;
  if (!registered) {
    std::atexit(shm_cleanup_all);
    std::signal(SIGINT, shm_signal_handler);
    std::signal(SIGTERM, shm_signal_handler);
    std::signal(SIGSEGV, shm_signal_handler);
    // SIGKILL cannot be caught
    registered = true;
  }
}
} // namespace detail

enum class SharedMemoryInit {
  OpenExisting,
  ExclusiveCreate,
};

struct SharedMemory {
  void *data = nullptr;
  size_t size = 0;
  size_t stride = 1;
  int fd = -1;
  std::string name;
  SharedMemoryInit mode;

  SharedMemory(SharedMemoryInit init, const std::string &name, size_t size,
               size_t stride = 1);
  ~SharedMemory();

  bool valid() const;

  int cleanup();

  // allow move
  SharedMemory(SharedMemory &&other) = default;
  SharedMemory &operator=(SharedMemory &&other) = default;

  // prevent copying
  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;

  template <typename T>
  T &at(size_t index);
  template <typename T>
  const T &at(size_t index) const;
};

template <typename T>
struct TypedSharedMemory : public SharedMemory {
  TypedSharedMemory(SharedMemoryInit init, const std::string &name,
                    size_t capacity);
  ~TypedSharedMemory();

  size_t capacity() const;
  T &operator[](size_t index);
  const T &operator[](size_t index) const;
};

template <typename T>
MOTR_ALWAYS_INLINE T &SharedMemory::at(size_t index) {
  assert(index < size / sizeof(T));
  return reinterpret_cast<T *>(data)[index];
}

template <typename T>
const T &SharedMemory::at(size_t index) const {
  assert(index < size / sizeof(T));
  return reinterpret_cast<const T *>(data)[index];
}

template <typename T>
TypedSharedMemory<T>::TypedSharedMemory(SharedMemoryInit init,
                                        const std::string &name,
                                        size_t capacity)
    : SharedMemory(init, name, sizeof(T) * capacity, sizeof(T)) {}

template <typename T>
TypedSharedMemory<T>::~TypedSharedMemory() {
  //
}

template <typename T>
MOTR_ALWAYS_INLINE size_t TypedSharedMemory<T>::capacity() const {
  return size / sizeof(T);
}

template <typename T>
MOTR_ALWAYS_INLINE T &TypedSharedMemory<T>::operator[](size_t index) {
  return at<T>(index);
}

template <typename T>
MOTR_ALWAYS_INLINE const T &
TypedSharedMemory<T>::operator[](size_t index) const {
  return at<T>(index);
}

} // namespace M

MOTR_ALWAYS_INLINE
M::SharedMemory::SharedMemory(SharedMemoryInit init, const std::string &_name,
                              size_t size, size_t stride)
    : data(nullptr),  //
      size(size),     //
      stride(stride), //
      fd(-1),         //
      name(_name),    //
      mode(init) {
  // MOTR_LOG("SHM init={} name={} size={} stride={}", int(init), name, size,
  // stride);
  [[maybe_unused]] int err = 0;
  if (name.size() > PSHMNAMLEN) {
    MOTR_LOG("SHM name={} is too long", name);
    err = 1;
    goto FAILURE;
    return;
  }

  switch (mode) {
  case SharedMemoryInit::OpenExisting:
    fd = shm_open(name.c_str(), O_RDWR, 0600);
    break;
  case SharedMemoryInit::ExclusiveCreate:
    // auto fd2 = shm_unlink(name.c_str());
    // MOTR_LOG("shm_open: name = '{}'", name);
    fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    break;
  }

  if (fd == -1) {
    // TODO: If exclusive create, check if the error is because the file already
    // exists if so, check control block to see if the process is active and
    // check timestamp and recreate it if it is not active or if the timestamp
    // is too old
    // MOTR_LOG("shm_open error: {} -> '{}'", errno, strerror(errno));
    err = 2;
    goto FAILURE;
  }

  if (mode == SharedMemoryInit::ExclusiveCreate && ftruncate(fd, size) == -1) {
    err = 3;
    goto FAILURE;
  }

  data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (data == MAP_FAILED) {
    data = nullptr;
    err = 4;
    goto FAILURE;
  }

  if (mode == SharedMemoryInit::ExclusiveCreate) {
    M::detail::shm_register_signal_handlers();
    {
      std::lock_guard<std::mutex> lock(M::detail::getShmRegistryMutex());
      M::detail::getShmRegistry().insert(name);
    }
    // zero out the memory
    // MOTR_LOG("Zeroing out SHM name={} size={}", name, size);
    // TODO: On Linux, this memset causes a segfault
#ifndef MOTR_PLATFORM_LINUX
    memset(data, 0, size);
#endif
  }

  // MOTR_LOG("SHM SUCCESS init={} name={} size={} fd={} data={:p}", int(init),
  // name, size, fd, data);

  return;

FAILURE:
  // MOTR_LOG("SHM FAILURE init={} name={} size={} fd={} data={:p} err={}",
  // int(init), name, size, fd, data, err); MOTR_LOG("errno={} errstr={}",
  // errno, strerror(errno));
  cleanup();
}

MOTR_ALWAYS_INLINE int M::SharedMemory::cleanup() {
  // MOTR_LOG(">SHM::cleanup mode={} name={} size={} fd={} data={:p}",
  // int(mode), name, size, fd, data);

  int err = 0;

  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, size);
    data = nullptr;
    err |= 0b001;
  }

  if (fd != -1) {
    close(fd);
    fd = -1;
    err |= 0b010;
  }

  if (mode == SharedMemoryInit::ExclusiveCreate) {
    if (shm_unlink(name.c_str()) != 0) {
      err |= 0b100;
    }
  }

  // MOTR_LOG("<SHM::cleanup err={:03b} mode={} name={} size={} fd={}
  // data={:p}", err, int(mode), name, size, fd, data);

  return err;
}

MOTR_ALWAYS_INLINE M::SharedMemory::~SharedMemory() {
  if (mode == SharedMemoryInit::ExclusiveCreate) {
    std::lock_guard<std::mutex> lock(M::detail::getShmRegistryMutex());
    M::detail::getShmRegistry().erase(name);
  }
  cleanup();
}

MOTR_ALWAYS_INLINE bool M::SharedMemory::valid() const {
  return data != nullptr && fd != -1;
}

#endif // MOTR_SHARED_MEMORY_H
