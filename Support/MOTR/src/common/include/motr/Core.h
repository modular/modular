//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_CORE_H
#define MOTR_CORE_H

#include "motr/Constants.h"
#include "motr/Hash.h"
#include "motr/Macros.h"
#include "motr/Mailbox.h"
#include "motr/Message.h"
#include "motr/Time.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>

#include <pthread.h>

namespace M::motr {

template <MessageType message_type, MessageFlags flags = MessageFlags::Atomic>
struct EmitMessage {

  using Mailbox = ServerOutbox;

  // default constructor automatically fetches parent id from the
  // top of the thread local stack
  EmitMessage();

  // constructor with explicit parent id does not push the message to the stack
  EmitMessage(uint64_t parent_id);

  // manually push the message to the stack,
  // and optionally pop the top of the stack when EmitMessage is destroyed
  EmitMessage &pushOnThreadLocalStack(bool popOnDestruction);

  // send the message to the global inbox queue
  EmitMessage &send();

  // on destruction, if not aready sent the message will be sent
  ~EmitMessage();
  Message msg;
  bool popOnDestruction = false;
};

MOTR_ALWAYS_INLINE uint64_t makeIdSafeForJavascript(uint64_t id) {
  // convert the uint64_t to a double
  // then back to uint64_t
  // because JSON only supports 52 bits of precision
  // the double will lose some precision, but it will be close
  double d = id;
  return d;
}

MOTR_ALWAYS_INLINE uint64_t getNextMessageID() {
  // Generate a random 64-bit id for the message
  // todo: this is in the hot path so profile this
  // to find a very fast, distributed random generator
  static thread_local std::mt19937_64 generator{std::random_device{}()};
  static thread_local std::uniform_int_distribution<uint64_t> distribution;
  return distribution(generator);
}

MOTR_ALWAYS_INLINE uint32_t getProcessID() {
  static const uint32_t procid = getpid();
  return procid;
}

// uint32_t getThreadID()
//
// Returns the thread id of the current thread
// 0 = main thread
// 1..n = non-main threads
// non-main thead ids are initialized on first use of the MOTR API
// not necessarily in the order of thread creation
MOTR_ALWAYS_INLINE uint32_t getThreadID() {
  // Note the internal accounting uses 1-based indexing
  //
  // 0 = uninitialized
  // 1 = main thread id
  // 2 = first non-main thread id
  static std::atomic<uint64_t> nonMainThreadCounter{0};
  thread_local uint64_t threadId = 0;
  if (threadId == 0) {
    // initialization runs once per thread

    if (
#if defined(MOTR_PLATFORM_LINUX)
        int(pthread_self()) == getpid()
#elif defined(MOTR_PLATFORM_MACOS)
        pthread_main_np() == 1
#elif defined(MOTR_PLATFORM_EMSCRIPTEN)
        pthread_self() == getpid()
#endif
    ) {
      threadId = 1;
    } else {
      threadId = nonMainThreadCounter.fetch_add(1);
      // 2 = first non-main thread id
      // so we add 2 to the non-main thread id
      threadId += 2;
    }
  }
  assert(threadId);

  // convert to 0-based indexing for the public API
  threadId -= 1;

  return threadId;
}

/// ProcessLifetime
///
/// Used to track the lifetime of the host process.
/// On construction, it sends out Message<Process, Push>
/// On destruction, it sends out Message<Process, Pop>
///
/// Singleton is created when:
/// The first Process message is sent...
/// which triggers the creation of a thread local ParentIdStack....
/// which triggers the creation of a process global ProcessLifetime singleton.
struct ProcessLifetime {
public:
  static uint64_t getSingletonMessageId() {
    return getProcessLifetimeSingleton().id;
  }

private:
  using Push = EmitMessage<MessageType::Process, MessageFlags::Push>;
  using Pop = EmitMessage<MessageType::Process, MessageFlags::Pop>;
  uint64_t id;
  ProcessLifetime() : id(Push{getProcessParentId()}.msg.id) {
    EmitMessage<MessageType::Set, MessageFlags::TagInt> tag{id};
    tag.msg.id = Constants::ProcessId::hash.v;
    tag.msg.ts = getProcessID();
  }
  ~ProcessLifetime() { Pop{id}; }

  static ProcessLifetime &getProcessLifetimeSingleton() {
    static ProcessLifetime processLifetimeSingleton;
    return processLifetimeSingleton;
  }

  static uint64_t getProcessParentId() {
    // TODO: get the parent id from the
    // environment var MODULAR_MOTR_ID
    // for now, just return 0
    return 0;
  }
};

/// ParentIdStack
///
/// Used to track the parent id of the current thread.
/// On construction, it sends out Message<Thread, Push>
/// On destruction, it sends out Message<Thread, Pop>
///
/// Thread local singleton is created when:
/// The first Thread message is sent...
/// which triggers the creation of a thread local ParentIdStack....
struct ParentIdStack {
  constexpr static uint64_t stackMaxSize = 1023;
  ParentIdStack();
  ~ParentIdStack();

  uint64_t stackSize = 0;
  uint64_t stack[stackMaxSize] = {0};

  MOTR_ALWAYS_INLINE void push(uint64_t id) {
    assert(stackSize < stackMaxSize);
    stack[stackSize++] = id;
  }

  MOTR_ALWAYS_INLINE uint64_t pop() {
    assert(stackSize > 0);
    return stack[--stackSize];
  }
  MOTR_ALWAYS_INLINE uint64_t top() const {
    assert(stackSize > 0);
    return stack[stackSize - 1];
  }

  using PushMsg = EmitMessage<MessageType::Thread, MessageFlags::Push>;
  using PopMsg = EmitMessage<MessageType::Thread, MessageFlags::Pop>;
};

MOTR_ALWAYS_INLINE ParentIdStack::ParentIdStack() {
  // This is the first time this thread is being used

  // Send a message to the server to register this thread span push
  PushMsg pushMsg{ProcessLifetime::getSingletonMessageId()};

  // Initialize this thread's ID stack to
  // reflect the current process id and thread id
  stackSize = 2;
  stack[0] = pushMsg.msg.pid; // same as getProcessLifetimeID();
  stack[1] = pushMsg.msg.id;

  {
    // Send a message to the server to register this thread id
    auto threadId = getThreadID();
    constexpr uint64_t threadIdStringHash = Constants::ThreadId::hash.v;
    ServerOutbox::send({
        MessageType::Set,     // type
        MessageFlags::TagInt, // flags
        {},                   // data
        0,                    // procid
        threadId,             // ts
        threadIdStringHash,   // id
        pushMsg.msg.id,       // parentid
    });
  }
}

MOTR_ALWAYS_INLINE ParentIdStack::~ParentIdStack() {
  // TODO: handle AsyncRT green threads
  assert(stackSize >= 2);
  if (stackSize > 2) {
    EmitMessage<MessageType::StackError> error;
    // The stack has more than 2 items,
    // something is wrong, but we can still unregister the thread
  }

  // Thread pop message is sent on scope exit
  PopMsg popMsg{stack[1]};
}

MOTR_ALWAYS_INLINE ParentIdStack &getThreadLocalParentIdStack() {
  // todo: handle AsyncRT green threads
  static thread_local ParentIdStack parentIdStack;
  return parentIdStack;
}

MOTR_ALWAYS_INLINE uint64_t getParentID() {
  // convenience funciton to get the top of the thread local stack
  return getThreadLocalParentIdStack().top();
}

MOTR_ALWAYS_INLINE void pushParentID(uint64_t id) {
  // convenience funciton to push to the thread local stack
  getThreadLocalParentIdStack().push(id);
}

MOTR_ALWAYS_INLINE uint64_t popParentID() {
  // convenience funciton to pop from the thread local stack
  auto id = getThreadLocalParentIdStack().pop();
  return id;
}

template <MessageType message_type, MessageFlags flags>
MOTR_ALWAYS_INLINE Message &initMessage(Message &msg, uint64_t parent_id) {
  msg.type = message_type;
  msg.flags = flags;
  msg.procid = getProcessID();
  msg.ts = nowNanoSeconds();
  msg.id = getNextMessageID();
  msg.pid = parent_id;
  return msg;
}

// default constructor gets the parent_id from the top of the thread local
// staack
template <MessageType message_type, MessageFlags flags>
MOTR_ALWAYS_INLINE EmitMessage<message_type, flags>::EmitMessage() {
  initMessage<message_type, flags>(msg, getParentID());
}

// explicit parent_id constrtuctor
template <MessageType message_type, MessageFlags flags>
MOTR_ALWAYS_INLINE
EmitMessage<message_type, flags>::EmitMessage(uint64_t parent_id) {
  initMessage<message_type, flags>(msg, parent_id);
}

template <MessageType message_type, MessageFlags flags>
MOTR_ALWAYS_INLINE EmitMessage<message_type, flags> &
EmitMessage<message_type, flags>::pushOnThreadLocalStack(
    bool _popOnDestruction) {
  assert(!popOnDestruction && "Message already pushed");
  pushParentID(msg.id);
  popOnDestruction = _popOnDestruction;
  return *this;
}

template <MessageType message_type, MessageFlags flags>
MOTR_ALWAYS_INLINE EmitMessage<message_type, flags> &
EmitMessage<message_type, flags>::send() {
  // todo: this should be a getGlobalInboxQueue
  // and init should happen in the first time the process push event is sent
  assert(msg.id != 0 && "Message id is 0, indicating it has already been sent");
  Mailbox::send(msg);
  msg.id = 0;
  return *this;
}

template <MessageType message_type, MessageFlags flags>
MOTR_ALWAYS_INLINE EmitMessage<message_type, flags>::~EmitMessage() {
  if (msg.id != 0)
    send();
  if (popOnDestruction)
    popParentID();
}

} // namespace M::motr

#endif
