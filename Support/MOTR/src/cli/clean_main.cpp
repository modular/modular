//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/Log.h"
#include "motr/motr.h"
#include <cstdlib>

template <typename T>
void cleanupSharedMemoryQueueConfig() {
  std::vector<std::pair<std::string, size_t>> names = {
      {M::motr::Namespace::join(T::name(), "ctrl"),
       sizeof(M::motr::detail::QueueControl)},
      {M::motr::Namespace::join(T::name(), "data"), T::totalMemorySize},
  };
  for (const auto &[name, size] : names) {
    bool open_valid = false;
    bool create_valid = false;
    bool expected_create_valid = true;
    {
      M::SharedMemory shm(M::SharedMemoryInit::OpenExisting, name, size);
      open_valid = shm.valid();
    }

    if (open_valid) {
      MOTR_LOG("motr clean found already open SHM: {} [size={}]", name, size,
               open_valid);
      expected_create_valid = false;
    }

    {
      M::SharedMemory shm(M::SharedMemoryInit::ExclusiveCreate, name, size);
      create_valid = shm.valid();
      int ret = shm.cleanup();

      if (create_valid != expected_create_valid) {
        MOTR_LOG("ERROR: motr clean {} unexpected valid={}", name,
                 create_valid);
      }

      MOTR_LOG("motr clean {} ret={:03b}", name, ret, create_valid);
    }

    {
      M::SharedMemory shm(M::SharedMemoryInit::OpenExisting, name, size);
      if (shm.valid()) {
        MOTR_LOG("ERROR: motr clean open again: {} size={}, valid={}\n", name,
                 size, shm.valid());
      }
    }

    // cleanup called on destruction
  }
  // for(const auto& name : names) {
  //   M::SharedMemory shm(M::SharedMemoryInit::OpenExisting, name,
  //   T::totalMemorySize); MOTR_LOG("SHM open: {} size={}, valid={}", name,
  //   T::totalMemorySize, shm.valid());
  // }
}

int cleanMain(int argc, char **argv) {
  // MOTR_Trace(clean);

  using namespace M::motr;

  std::string originalNamespace = Namespace::get();
  std::string_view cleanupNamespace = originalNamespace;
  MOTR_LOG("cleanup starting namespace in {}={}", Namespace::EnvVar,
           originalNamespace);

  if (argc > 2) {
    std::string_view namespaceSv = argv[2];
    MOTR_LOG("cleanup {}={}", Namespace::EnvVar, namespaceSv);
    Namespace::set(namespaceSv);
  }

  cleanupSharedMemoryQueueConfig<ServerInbox::Config>();
  cleanupSharedMemoryQueueConfig<ServerInboxString::Config>();

  if (originalNamespace != cleanupNamespace) {
    Namespace::set(originalNamespace);
  } else {
    MOTR_LOG("motr clean queue pointer reset namespace={} queue pointers",
             originalNamespace);

    ServerInbox::getQueuePtrRef().reset();
    ServerOutbox::getQueuePtrRef().reset();

    ServerInboxString::getQueuePtrRef().reset();
    ServerOutboxString::getQueuePtrRef().reset();
  }
  MOTR_LOG("motr clean namespace={} done", cleanupNamespace);

  if (false) {
    const auto name = M::motr::Flags::detail::getSharedMemoryName();
    const auto size = M::motr::Flags::Manager::numBytes;
    M::SharedMemory shm(M::SharedMemoryInit::ExclusiveCreate, name, size);
  }

  return 0;
}
