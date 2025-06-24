//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/Log.h"
#include "motr/motr.h"
#include <thread>

using namespace M;

int testMain(int argc, char **argv) {
  using Outbox = motr::ServerOutbox;
  if (!Outbox::valid()) {
    MOTR_LOG("Server outbox is not valid\n", "");
    return 2;
  }

  using StringOutbox = motr::ServerOutboxString;
  StringOutbox stringOutbox;
  if (!stringOutbox.valid()) {
    MOTR_LOG("Server string outbox is not valid\n", "");
    return 2;
  }

  MOTR_Trace(messageTestTrace);

  static uint64_t counter = 0;
  for (int i = 0; i < 10; i++) {
    MOTR_Trace(loop_span);
    MOTR_TagStrOnce("key", "10");

    motr::SendTags{{
        {"key", "10"},
        {"key2", "20"},
        {"key3", "30"},
        {"key4", "40"},
        {"key5", "50"},
        {"key6", "60"},
        {"key7", "70"},
        {"key8", "80"},
        {"key9", "90"},
        {"key10", "100"},
    }};

    std::vector<motr::Message> data;
    for (int j = 0; j < i; j++) {
      ++counter;
      auto msg_type = static_cast<motr::MessageType>(
          counter % int(motr::MessageType::COUNT));
      if (msg_type == motr::MessageType::Reload)
        continue;
      if (msg_type == motr::MessageType::Stop)
        continue;
      MOTR_LOG("Sending counter={}", counter);
      MOTR_LOG("msg_type={}", motr::toString(msg_type));
      auto timestamp = M::motr::nowNanoSeconds();
      auto id = counter;
      auto parent_id = 10000 + id;

      motr::Message msg{msg_type,
                        motr::MessageFlags::None,
                        {},
                        motr::getProcessID(),
                        uint64_t(timestamp),
                        id,
                        parent_id};

      data.emplace_back(msg);
    }
    MOTR_LOG("Sending {} items...", data.size());
    Outbox::send(data.data(), data.size());

    StringOutbox::send({"Hello, world!"});
    StringOutbox::send({fmt::format("Iteration {}", i)});

    int sleepTime = 1000;
    MOTR_LOG("sleeping for {}ms...", sleepTime);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
  }
  MOTR_LOG("sending Stop message\n", "");
  motr::EmitMessage<motr::MessageType::Stop>{};
  MOTR_LOG("Done sending items\n", "");
  return 0;
}
