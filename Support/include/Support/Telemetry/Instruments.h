//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_INSTRUMENTS_H
#define SUPPORT_TELEMETRY_INSTRUMENTS_H

namespace M::Telemetry {

class TelemetryContext;

template <typename T>
class Counter {
public:
  void add(T value) {}

private:
  friend class TelemetryContext;

  // TODO: add arguments, depending on how Counter is implemented.
  Counter() {}
};

template <typename T>
class Histogram {
public:
  void record(T value) {}

private:
  friend class TelemetryContext;

  // TODO: add arguments, depending on how Histogram is implemented.
  Histogram() {}
};

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_INSTRUMENTS_H
