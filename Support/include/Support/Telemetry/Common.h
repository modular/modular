//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_COMMON_H
#define SUPPORT_TELEMETRY_COMMON_H

namespace M::Telemetry {

/// Telemetry levels. We emit more information with increasing telemetry level,
/// such that if the configured telemetry level is X, we emit all signals
/// (metrics, logs) tagged with level <= X.
enum class Level : uint8_t { L0, L1, L2, USER = 255 };

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_COMMON_H
