//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_LOGCHANNELS_H
#define SUPPORT_LOGCHANNELS_H

#include <bitset>
#include <climits>
#include <cstdint>

namespace M::Log {

// Defines all possible channels for the Mojo logger. To add a new channel,
// add a new `X` row with the enum member name and its config entry.
#define MLOG_CHANNELS(X)                                                       \
  X(Default, "default")                                                        \
  X(Mojo, "mojo")                                                              \
  X(MLRT, "mlrt")

namespace Channel {
enum Channels : uint64_t {
#define MEMBER_NAME(channel, a_) channel,
  MLOG_CHANNELS(MEMBER_NAME)
#undef MEMBER_NAME
      NChannels
};
} // namespace Channel

// Controls per-channel output toggles for the channels in the logger.
class ChannelState {
  static constexpr size_t bitCount = sizeof(uint64_t) * CHAR_BIT;
  static_assert(Channel::NChannels <= bitCount);
  std::bitset<bitCount> enabled;

public:
  ChannelState() { enabled[Channel::Default] = true; }

  void enable(Channel::Channels c) { enabled[c] = true; }

  void enableAll() { enabled.set(); }

  void disable(Channel::Channels c) { enabled[c] = false; }

  void disableAll() { enabled.reset(); }

  bool isEnabled(Channel::Channels c) const { return enabled[c]; }
};

} // namespace M::Log

#endif // SUPPORT_LOGCHANNELS_H
