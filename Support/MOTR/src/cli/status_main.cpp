//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/Log.h"
#include "motr/motr.h"

using namespace M;

int statusMain(int argc, char **argv) {
  MOTR_Trace(status);
  MOTR_LOG("GlobalServerOutbox: {}", motr::ServerOutbox::valid());
  if (motr::ServerOutbox::valid()) {
    M::motr::ServerOutbox::getQueue().debugPrint();
  }
  MOTR_LOG("GlobalServerOutboxString: {}", motr::ServerOutboxString::valid());
  if (motr::ServerOutboxString::valid()) {
    M::motr::ServerOutboxString::getStringQueue().control.debugPrint(0);
  }
  auto &shm = M::motr::Flags::Manager::getSharedMemory();
  MOTR_LOG("Flags SharedMemory.valid: {}", shm.valid());
  MOTR_LOG("Flags SharedMemory.size: {}", shm.size);
  MOTR_LOG("Flags SharedMemory.data: {}", shm.data);
  MOTR_LOG("Flags SharedMemory.fd: {}", shm.fd);
  MOTR_LOG("Flags SharedMemory.mode: {}", int(shm.mode));
  MOTR_LOG("Flags SharedMemory.name: {}", shm.name);
  return 0;
}
