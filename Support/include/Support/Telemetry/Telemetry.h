//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_H
#define SUPPORT_TELEMETRY_H

#include "Support/Telemetry/Instruments.h"

namespace M::Telemetry {

// TODO: Add ways to organize instruments (e.g. Meters/instrumentation scope)
// later if needed.

/// A TelemetryContext provides access to instruments (e.g. Counter, Histogram)
/// to instrument the code and generate metrics. These metrics will be exported
/// by the TelemetryContext based on the options passed to it during creation.
///
/// Right now we are assuming that the TelemetryContext will collect Resource
/// attributes (e.g. CPU info, OS info, version of software components) without
/// this information being passed to it explicitly through its API, but this is
/// subject to change.
class TelemetryContext {
public:
  // TODO: add options, like exporter options (HTTP URL, file name).
  TelemetryContext() = default;

  // XXX: not sure if it's better to allocate Counter and Histogram on the heap
  // or not. For Otel, the Counter struct will basically just contain a pointer
  // to the Otel counter, and so returning the struct seems appropiate.

  /// Create a Counter<uint64_t>.
  Counter<uint64_t> createUInt64Counter(StringRef name,
                                        StringRef description = "",
                                        StringRef unit = "") {
    // TODO: If the name is invalid, it looks like OTel logs the error and
    // returns a NOOP counter. Instead, we should probably try to assert that
    // the name is valid or that the returned counter is not NOOP. Same for
    // other instruments.
    return Counter<uint64_t>();
  }

  /// Create a Counter<double>.
  Counter<double> createDoubleCounter(StringRef name,
                                      StringRef description = "",
                                      StringRef unit = "") {
    return Counter<double>();
  }

  /// Create a Histogram<uint64_t>.
  Histogram<uint64_t> createUInt64Histogram(StringRef name,
                                            StringRef description = "",
                                            StringRef unit = "") {
    return Histogram<uint64_t>();
  }

  /// Create a Histogram<double>.
  Histogram<double> createDoubleHistogram(StringRef name,
                                          StringRef description = "",
                                          StringRef unit = "") {
    return Histogram<double>();
  }
};

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_H
