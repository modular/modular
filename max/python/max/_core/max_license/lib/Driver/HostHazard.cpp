//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "_core/Driver/HostHazard.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "_core/Driver/Device.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace M;
using namespace Driver;

//===----------------------------------------------------------------------===//
// HostHazardCompletion
//===----------------------------------------------------------------------===//

uintptr_t Python::HostHazardCompletion::queueKey() const {
  return reinterpret_cast<uintptr_t>(queue->stream->ctx.getPointer());
}

bool Python::HostHazardCompletion::isReady() const {
  if (!producerError.empty() || !event)
    return true;
  return event->isReady();
}

void Python::HostHazardCompletion::wait() const {
  if (!producerError.empty())
    throw HostHazardError(
        "Attempting to wait on a completion with a producer error, any "
        "associated Buffers cannot be written to or read from by the host "
        "until successfully written to by the device:\n" +
        producerError);
  if (!event)
    return;

  // `DeviceEvent::synchronize` already releases the GIL
  event->synchronize();
}

void Python::HostHazardCompletion::poison(const std::string &msg) {
  producerError = msg;
}

std::shared_ptr<Python::HostHazardCompletion>
Python::HostHazardCompletion::recordOn(Python::PyDeviceStreamTy queue) {
  // Doesn't release the GIL: nesting two releases aborts.
  auto eventOr = queue->stream->ctx->enqueueEvent();
  if (eventOr.isError())
    throw std::runtime_error(eventOr.takeError().get());

  auto out = std::make_shared<HostHazardCompletion>();
  out->event = std::make_shared<Python::DeviceEvent>(eventOr.takeValue());
  out->queue = std::move(queue);
  return out;
}

std::shared_ptr<Python::HostHazardCompletion>
Python::HostHazardCompletion::poisoned(Python::PyDeviceStreamTy queue,
                                       const std::string &error) {
  auto out = std::make_shared<HostHazardCompletion>();
  out->event = nullptr;
  out->queue = std::move(queue);
  out->poison(error);

  return out;
}

//===----------------------------------------------------------------------===//
// HostHazardRecord
//===----------------------------------------------------------------------===//

void Python::HostHazardRecord::stampWrite(
    std::shared_ptr<HostHazardCompletion> completion) {
  std::lock_guard<std::mutex> lock(mu);
  // Only this queue's read is ordered behind the write; foreign ones survive.
  readers.erase(completion->queueKey());
  lastWrite = std::move(completion);
}

void Python::HostHazardRecord::stampRead(
    std::shared_ptr<HostHazardCompletion> completion) {
  std::lock_guard<std::mutex> lock(mu);
  readers[completion->queueKey()] = std::move(completion);
}

void Python::HostHazardRecord::taint() {
  std::lock_guard<std::mutex> lock(mu);
  tainted = true;
}

bool Python::HostHazardRecord::isTainted() const {
  std::lock_guard<std::mutex> lock(mu);
  return tainted;
}

void Python::HostHazardRecord::clearTaint() {
  std::lock_guard<std::mutex> lock(mu);
  tainted = false;
}

std::shared_ptr<Python::HostHazardCompletion>
Python::HostHazardRecord::takeLastWriteSnapshot() {
  std::lock_guard<std::mutex> lock(mu);
  return lastWrite;
}

std::vector<std::shared_ptr<Python::HostHazardCompletion>>
Python::HostHazardRecord::takeReadersSnapshot() {
  std::lock_guard<std::mutex> lock(mu);

  std::vector<std::shared_ptr<HostHazardCompletion>> ret;
  ret.reserve(readers.size());

  for (const auto &kv : readers) {
    ret.push_back(kv.second);
  }
  return ret;
}

void Python::HostHazardRecord::clearSettled(
    std::shared_ptr<HostHazardCompletion> write,
    std::vector<std::shared_ptr<HostHazardCompletion>> readers) {

  if (write && !write->isReady())
    write = nullptr;
  for (auto &reader : readers) {
    if (reader && !reader->isReady())
      reader = nullptr;
  }

  std::lock_guard<std::mutex> lock(mu);

  auto completeAndSame = [](const auto &l, const auto &r) -> bool {
    return l && l.get() == r.get();
  };

  if (completeAndSame(write, lastWrite))
    lastWrite = nullptr;
  for (const auto &reader : readers) {
    // Nulled above when still pending.
    if (!reader)
      continue;

    uintptr_t qk = reader->queueKey();
    if (!this->readers.contains(qk))
      continue;
    if (completeAndSame(reader, this->readers[qk]))
      this->readers.erase(qk);
  }
}

//===----------------------------------------------------------------------===//
// HostHazardError
//===----------------------------------------------------------------------===//

Python::HostHazardError::HostHazardError(std::string msg)
    : message(std::move(msg)) {}

const char *Python::HostHazardError::what() const noexcept {
  return message.c_str();
}

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

std::unique_ptr<Python::HostHazardRecord>
Python::cloneHostHazardRecord(const HostHazardRecord *record) {
  if (!record)
    return nullptr;
  auto out = std::make_unique<HostHazardRecord>();
  // Locked: prevent edits to source mid flight
  std::lock_guard<std::mutex> guard(record->mu);
  out->lastWrite = record->lastWrite;
  out->readers = record->readers;
  out->tainted = record->tainted;
  return out;
}
