//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_SPAN_H
#define MOTR_SPAN_H

#include "motr/Core.h"
#include "motr/Macros.h"
#include "motr/Message.h"

namespace M::motr {

using EmitSpanPushMessage = EmitMessage<MessageType::Span, MessageFlags::Push>;
using EmitSpanPopMessage = EmitMessage<MessageType::Span, MessageFlags::Pop>;
using EmitMarkMessage = EmitMessage<MessageType::Status, MessageFlags::Atomic>;

// Span pairs together a push and pop message using RAII
template <MessageType message_type>
struct Span;

using TraceSpan = Span<MessageType::Span>;

// Span pairs together a push and pop message using RAII
template <MessageType message_type>
struct Span {
  Span();
  Span(uint64_t parent_id);
  ~Span();

  Span(const Span &) = delete;
  Span &operator=(const Span &) = delete;
  Span(Span &&) = delete;
  Span &operator=(Span &&) = delete;
};

template <MessageType message_type>
MOTR_ALWAYS_INLINE Span<message_type>::Span() {
  EmitMessage<message_type, MessageFlags::Push> push;
  // manually push the id of the Span push message onto the stack
  // so all subsequent messages will be in the scope of the Span
  // until the Span is destroyed
  push.pushOnThreadLocalStack(false);
}

template <MessageType message_type>
MOTR_ALWAYS_INLINE Span<message_type>::Span(uint64_t parent_id) {
  EmitMessage<message_type, MessageFlags::Push> push(parent_id);
  // manually push the id of the Span push message onto the stack
  // so all subsequent messages will be in the scope of the Span
  // until the Span is destroyed
  push.pushOnThreadLocalStack(false);
}

template <MessageType message_type>
MOTR_ALWAYS_INLINE Span<message_type>::~Span() {
  // Pop the top scope off the stack
  uint64_t parentId = popParentID();
  // todo: debug mode should assert that the
  // parentId is the equal to Span::Span push.msg.id

  // Emmit event that the Span is ending
  EmitMessage<message_type, MessageFlags::Pop> pop(parentId);
}

} // namespace M::motr

#endif
