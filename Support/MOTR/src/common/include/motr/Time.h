//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_TIME_H
#define MOTR_TIME_H

#include "motr/Log.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <ctime>
#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "fmt/ranges.h"
#include <chrono>
#include <string>
#include <thread>

#include <time.h>

#include "motr/Macros.h"

namespace M::motr {
[[nodiscard]] int64_t parseTime(const char *timeStr);
[[nodiscard]] int64_t parseDate(const char *dateStr);
[[nodiscard]] int64_t getBuildTimestampInSeconds();
[[nodiscard]] int64_t getElapsedSecondsSinceBuild();
[[nodiscard]] std::string secondsToTimeString(uint64_t seconds);
[[nodiscard]] std::string timeNsToISODate(int64_t nanoseconds);
[[nodiscard]] int64_t nowNanoSeconds();
} // namespace M::motr

namespace M::motr::Time {

enum class TimeType { Timestamp, Duration };

template <TimeType type>
struct Nanoseconds;

struct Timestamp;
struct Duration;

static inline Timestamp now();

Timestamp getBuildTimestamp();

// Must be called at startup to initialize the start timestamp
[[nodiscard]] const Timestamp &getStartTimestamp();

enum class Precision {
  Nanoseconds,
  Microseconds,
  Milliseconds,
  Seconds,
  Minutes,
  Hours,
  Days,
  Weeks,
  Months,
  Years,
};

constexpr int64_t nanosecondsPer(Precision precision) {
  switch (precision) {
  case Precision::Nanoseconds:
    return 1;
  case Precision::Microseconds:
    return 1'000;
  case Precision::Milliseconds:
    return 1'000'000;
  case Precision::Seconds:
    return 1'000'000'000;
  case Precision::Minutes:
    return 60'000'000'000;
  case Precision::Hours:
    return 3'600'000'000'000;
  case Precision::Days:
    return 86'400'000'000'000;
  case Precision::Weeks:
    return 604'800'000'000'000;
  case Precision::Months:
    return 2'629'746'000'000'000;
  case Precision::Years:
    // exactly 365.2425 days
    return 31'556'952'000'000'000;
  }
  return 1;
}

constexpr const char *precisionSuffix(Precision precision) {
  switch (precision) {
  case Precision::Nanoseconds:
    return "ns";
  case Precision::Microseconds:
    return "us";
  case Precision::Milliseconds:
    return "ms";
  case Precision::Seconds:
    return "s";
  case Precision::Minutes:
    return "m";
  case Precision::Hours:
    return "h";
  case Precision::Days:
    return "d";
  case Precision::Weeks:
    return "w";
  case Precision::Months:
    return "mo";
  case Precision::Years:
    return "y";
  }
  return "";
}

template <Precision precision>
constexpr int64_t nanosecondsPer() {
  return nanosecondsPer(precision);
}

template <TimeType type>
struct Nanoseconds {
  int64_t v = 0;
  Nanoseconds(int64_t v) : v(v) {}
  int64_t nanoseconds() const { return v; }
  int64_t microseconds() const {
    return v / nanosecondsPer<Precision::Microseconds>();
  }
  int64_t milliseconds() const {
    return v / nanosecondsPer<Precision::Milliseconds>();
  }
  int64_t seconds() const { return v / nanosecondsPer<Precision::Seconds>(); }
  int64_t minutes() const { return v / nanosecondsPer<Precision::Minutes>(); }
  int64_t hours() const { return v / nanosecondsPer<Precision::Hours>(); }
  int64_t days() const { return v / nanosecondsPer<Precision::Days>(); }
  int64_t weeks() const { return v / nanosecondsPer<Precision::Weeks>(); }
  int64_t months() const { return v / nanosecondsPer<Precision::Months>(); }
  int64_t years() const { return v / nanosecondsPer<Precision::Years>(); }

  int64_t to(Precision precision) const {
    switch (precision) {
    case Precision::Nanoseconds:
      return nanoseconds();
    case Precision::Microseconds:
      return microseconds();
    case Precision::Milliseconds:
      return milliseconds();
    case Precision::Seconds:
      return seconds();
    case Precision::Minutes:
      return minutes();
    case Precision::Hours:
      return hours();
    case Precision::Days:
      return days();
    case Precision::Weeks:
      return weeks();
    case Precision::Months:
      return months();
    case Precision::Years:
      return years();
    }
  }
};

struct Timestamp : public Nanoseconds<TimeType::Timestamp> {
  using Base = Nanoseconds<TimeType::Timestamp>;
  using Base::Base;
  Timestamp() : Base(0) {}
  Timestamp(int64_t v) : Base(v) {}
  static Timestamp now() { return {nowNanoSeconds()}; }
  std::string toISO8601String() const;
  std::string toString(Precision precision = Precision::Seconds) const;
  static Timestamp min() { return {std::numeric_limits<int64_t>::min()}; }
  static Timestamp max() { return {std::numeric_limits<int64_t>::max()}; }
};

struct Duration : public Nanoseconds<TimeType::Duration> {
  using Base = Nanoseconds<TimeType::Duration>;
  using Base::Base;
  Duration() : Base(0) {}
  Duration(int64_t v) : Base(v) {}
  static Duration max() { return {std::numeric_limits<int64_t>::max()}; }
  std::string toString(Precision precision = Precision::Seconds) const;
  static Duration fromSeconds(double seconds) {
    return Duration{static_cast<int64_t>(seconds * 1'000'000'000)};
  }
  static Duration fromMilliseconds(double milliseconds) {
    return Duration{static_cast<int64_t>(milliseconds * 1'000'000)};
  }

  void sleep() const {
    std::this_thread::sleep_for(std::chrono::nanoseconds(v));
  }
};

// Timestamp and Duration arithmetic
MOTR_ALWAYS_INLINE Timestamp operator+(const Timestamp &lhs,
                                       const Duration &rhs) {
  return {lhs.v + rhs.v};
}

MOTR_ALWAYS_INLINE Timestamp operator-(const Timestamp &lhs,
                                       const Duration &rhs) {
  return {lhs.v - rhs.v};
}

MOTR_ALWAYS_INLINE Duration operator-(const Timestamp &lhs,
                                      const Timestamp &rhs) {
  return {lhs.v - rhs.v};
}

MOTR_ALWAYS_INLINE Duration operator+(const Duration &lhs,
                                      const Duration &rhs) {
  return {lhs.v + rhs.v};
}

MOTR_ALWAYS_INLINE Duration operator-(const Duration &lhs,
                                      const Duration &rhs) {
  return {lhs.v - rhs.v};
}

// Timestamp comparison operators
MOTR_ALWAYS_INLINE bool operator<(const Timestamp &lhs, const Timestamp &rhs) {
  return lhs.v < rhs.v;
}

MOTR_ALWAYS_INLINE bool operator>(const Timestamp &lhs, const Timestamp &rhs) {
  return lhs.v > rhs.v;
}

MOTR_ALWAYS_INLINE bool operator<=(const Timestamp &lhs, const Timestamp &rhs) {
  return lhs.v <= rhs.v;
}

MOTR_ALWAYS_INLINE bool operator>=(const Timestamp &lhs, const Timestamp &rhs) {
  return lhs.v >= rhs.v;
}

MOTR_ALWAYS_INLINE bool operator==(const Timestamp &lhs, const Timestamp &rhs) {
  return lhs.v == rhs.v;
}

MOTR_ALWAYS_INLINE bool operator!=(const Timestamp &lhs, const Timestamp &rhs) {
  return lhs.v != rhs.v;
}

// Duration comparison operators
MOTR_ALWAYS_INLINE bool operator<(const Duration &lhs, const Duration &rhs) {
  return lhs.v < rhs.v;
}

MOTR_ALWAYS_INLINE bool operator>(const Duration &lhs, const Duration &rhs) {
  return lhs.v > rhs.v;
}

MOTR_ALWAYS_INLINE bool operator<=(const Duration &lhs, const Duration &rhs) {
  return lhs.v <= rhs.v;
}

MOTR_ALWAYS_INLINE bool operator>=(const Duration &lhs, const Duration &rhs) {
  return lhs.v >= rhs.v;
}

MOTR_ALWAYS_INLINE bool operator==(const Duration &lhs, const Duration &rhs) {
  return lhs.v == rhs.v;
}

MOTR_ALWAYS_INLINE bool operator!=(const Duration &lhs, const Duration &rhs) {
  return lhs.v != rhs.v;
}

struct Range {
  Timestamp start = Timestamp::max();
  Timestamp end = Timestamp::min();
  bool valid() const { return start.v <= end.v; }
  Duration duration() const { return end - start; }
  void add(const Timestamp &ts) {
    if (ts < start)
      start = ts;
    if (ts > end)
      end = ts;
  }
  void add(const Range &other) {
    add(other.start);
    add(other.end);
  }

  void intersect(const Range &other) {
    if (other.start > start)
      start = other.start;
    if (other.end < end)
      end = other.end;
  }

  double scale(Precision precision = Precision::Nanoseconds) const {
    return 1.0 / duration().to(precision);
  }

  double bias(Precision precision = Precision::Nanoseconds) const {
    return -start.to(precision) * scale(precision);
  }
};

struct Elapsed {
  Timestamp t0;
  Timestamp t1;
  Elapsed() : t0(Timestamp::now()), t1(t0) {}
  const Timestamp &mark() {
    t1 = Timestamp::now();
    return t1;
  }
  Duration elapsed() { return mark() - t0; }
  std::string toString() { return elapsed().toString(); }
};

static inline Timestamp now() { return Timestamp::now(); }

} // namespace M::motr::Time

namespace M::motr {

#ifdef __EMSCRIPTEN__

namespace detail {
extern "C" {
double emscripten_date_now(void);
double emscripten_performance_now(void);
}
} // namespace detail

inline int64_t nowNanoSeconds() {
  static int64_t offset = 0;
  constexpr uint64_t nsToMs = 1'000'000;
  int64_t ts = detail::emscripten_performance_now() * nsToMs;
  if (offset == 0) {
    offset = detail::emscripten_date_now() * nsToMs;
    offset = offset - ts;
    assert(offset >= 0);
  }
  ts = ts + offset;
  return ts;
}
#endif

#ifdef __APPLE__

inline int64_t nowNanoSeconds() {
  static int64_t offset = 0;
  int64_t ns = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
  if (offset == 0) {
    offset = clock_gettime_nsec_np(CLOCK_REALTIME);
    offset = offset - ns;
  }
  ns = ns + offset;
  return ns;
}

#endif

#ifdef __linux__

inline int64_t nowNanoSeconds() {
  static int64_t offset = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  int64_t ns = ts.tv_sec * 1'000'000'000 + ts.tv_nsec;
  if (offset == 0) {
    clock_gettime(CLOCK_REALTIME, &ts);
    offset = ts.tv_sec * 1'000'000'000 + ts.tv_nsec - ns;
  }
  ns += offset;
  return ns;
}

#endif

struct DateTime {
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
  int nanosecond;

  static DateTime fromNanoseconds(int64_t nanoseconds);
  int64_t toNanoseconds() const;
  std::string toISOString() const;
  std::string toDebugString() const;
  std::string toFilenameString(std::string_view separator = "-") const;
};

inline DateTime DateTime::fromNanoseconds(int64_t nanoseconds) {
  assert(nanoseconds >= 0 && "Dates before unix epoch are not supported");

  constexpr int64_t NANOSECONDS_IN_SECOND = 1'000'000'000;
  constexpr int64_t SECONDS_IN_MINUTE = 60;
  constexpr int64_t MINUTES_IN_HOUR = 60;
  constexpr int64_t HOURS_IN_DAY = 24;
  constexpr uint64_t DAYS_IN_YEAR = 365;

  int64_t totalSeconds = nanoseconds / NANOSECONDS_IN_SECOND;
  int64_t fractionalSeconds = nanoseconds % NANOSECONDS_IN_SECOND;

  int64_t seconds = totalSeconds % SECONDS_IN_MINUTE;
  int64_t totalMinutes = totalSeconds / SECONDS_IN_MINUTE;
  int64_t minutes = totalMinutes % MINUTES_IN_HOUR;
  int64_t totalHours = totalMinutes / MINUTES_IN_HOUR;
  int64_t hours = totalHours % HOURS_IN_DAY;
  int64_t totalDays = totalHours / HOURS_IN_DAY;

  auto isLeapYear = [](int year) constexpr {
    return (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
  };

  auto getYear = [&isLeapYear](int64_t &totalDays) {
    int year = 1970;
    while (true) {
      int64_t daysInYear = DAYS_IN_YEAR + (isLeapYear(year) ? 1 : 0);
      if (totalDays < daysInYear)
        break;
      totalDays -= daysInYear;
      ++year;
    }
    return year;
  };

  auto getMonth = [&isLeapYear](int64_t &totalDays, int year) {
    static const int daysInMonthNormal[] = {31, 28, 31, 30, 31, 30,
                                            31, 31, 30, 31, 30, 31};
    static const int daysInMonthLeap[] = {31, 29, 31, 30, 31, 30,
                                          31, 31, 30, 31, 30, 31};

    const int *daysInMonth =
        isLeapYear(year) ? daysInMonthLeap : daysInMonthNormal;

    int month = 0;
    while (totalDays >= daysInMonth[month]) {
      totalDays -= daysInMonth[month];
      ++month;
    }
    return month;
  };

  int year = getYear(totalDays);
  int month = getMonth(totalDays, year);

  int day = static_cast<int>(totalDays); // Days are 1-based
  int hour = static_cast<int>(hours);
  int minute = static_cast<int>(minutes);
  int second = static_cast<int>(seconds);
  int nanosecond = static_cast<int>(fractionalSeconds);

  return DateTime{year, month, day, hour, minute, second, nanosecond};
}

inline std::string DateTime::toISOString() const {
  return fmt::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z", year,
                     month + 1, day + 1, // Months are 0-based
                     hour, minute, second, nanosecond / 1'000);
}

inline std::string
DateTime::toFilenameString(std::string_view separator) const {
  std::vector<std::string> parts = {
      fmt::format("{:04}", year),    fmt::format("{:02}", month + 1),
      fmt::format("{:02}", day + 1), fmt::format("{:02}", hour),
      fmt::format("{:02}", minute),  fmt::format("{:02}", second),
  };
  std::string sep{separator};
  return fmt::format("{}", fmt::join(parts, sep));
}

inline std::string DateTime::toDebugString() const {
  return fmt::format("year={}\nmonth={}\nday={}\nhour={}\nminute={}\nsecond={"
                     "}\nnanosecond={}",
                     year, month, day, hour, minute, second, nanosecond);
}

inline std::string timeNsToISODate(int64_t nanoseconds) {
  if (nanoseconds < 0) {
    return "<invalid>";
  }
  DateTime dt = DateTime::fromNanoseconds(nanoseconds);
  return dt.toISOString();
}

} // namespace M::motr

namespace M::motr {

// parses the preprocessor __TIME__ macro
// returns the number of seconds since midnight
inline int64_t parseTime(const char *timeStr) {
  int hour, min, sec;
  sscanf(timeStr, "%d:%d:%d", &hour, &min, &sec);
  return hour * 3600 + min * 60 + sec;
}

// parses the preprocessor __DATE__ macro
// returns the number of seconds since epoch
inline int64_t parseDate(const char *dateStr) {
  static const char *months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

  char month[4];
  int day, year;
  sscanf(dateStr, "%s %d %d", month, &day, &year);

  int monthNum = 0;
  for (int i = 0; i < 12; i++)
    if (strcmp(month, months[i]) == 0) {
      monthNum = i;
      break;
    }

  struct tm timeinfo = {};
  timeinfo.tm_sec = 0;
  timeinfo.tm_min = 0;
  timeinfo.tm_hour = 0;
  timeinfo.tm_mday = day;
  timeinfo.tm_mon = monthNum;
  timeinfo.tm_year = year - 1900;
  timeinfo.tm_isdst = -1;
  // https://cplusplus.com/reference/ctime/mktime/
  int64_t timestamp = mktime(&timeinfo);
  return timestamp;
}

inline std::string secondsToTimeString(uint64_t seconds) {
  uint64_t hours = seconds / 3600;
  uint64_t minutes = (seconds % 3600) / 60;
  uint64_t remaining_seconds = seconds % 60;
  std::string str;
  if (hours > 0)
    str += fmt::format("{}h", hours);
  if (minutes > 0 || !str.empty())
    str += fmt::format("{:2d}m", minutes);
  str += fmt::format("{:2d}s", remaining_seconds);
  return str;
}

} // namespace M::motr

inline const M::motr::Time::Timestamp &M::motr::Time::getStartTimestamp() {
  static Timestamp startTimestamp = Timestamp::now();
  return startTimestamp;
}

inline int64_t M::motr::getElapsedSecondsSinceBuild() {
  int64_t build_timestamp = getBuildTimestampInSeconds();
  Time::Timestamp now = M::motr::Time::Timestamp::now();
  int64_t now_seconds = now.seconds();
  auto elapsed = now_seconds - build_timestamp;
  return elapsed;
}

// returns build time in seconds since epoch
inline int64_t M::motr::getBuildTimestampInSeconds() {
  static int64_t build_timestamp = []() -> int64_t {
    int64_t date_timestamp = parseDate(__DATE__);
    int64_t time_timestamp = parseTime(__TIME__);
    return date_timestamp + time_timestamp;
  }();
  return build_timestamp;
}

inline M::motr::Time::Timestamp M::motr::Time::getBuildTimestamp() {
  // convert to nanoseconds
  return M::motr::Time::Timestamp{getBuildTimestampInSeconds() * 1'000'000'000};
}

namespace M::motr::Time {
// not used at the moment
template <int NUMERATOR, int POWER>
struct TimeUnitPerSecond {
  static constexpr double pow10(int power) {
    double result = 1.0;
    for (; power < 0; power++)
      result *= 1e-10;
    for (; power > 0; power--)
      result *= 1e10;
    return result;
  }
  // negative power means per second
  static constexpr double denominator = pow10(-POWER);
  static constexpr double numerator = NUMERATOR;
  static constexpr double value = NUMERATOR * denominator;
};

inline std::string getElapsedTimeString(M::motr::Time::Timestamp timestamp) {
  static M::motr::Time::Timestamp programStart =
      M::motr::Time::Timestamp::now();
  M::motr::Time::Duration elapsed = timestamp - programStart;
  return elapsed.toString(Precision::Seconds);
}

inline std::string Timestamp::toString(Precision precision) const {
  if (v < 0)
    return "<invalid>";
  DateTime dt = DateTime::fromNanoseconds(v);
  return dt.toISOString();
}

inline std::string Duration::toString(Precision max_precision) const {
  int64_t dur = nanoseconds();
  if (dur == 0)
    return "0s";
  bool isNegative = dur < 0;
  if (isNegative)
    dur = -dur;

  std::string result;

  auto extract = [&dur](Precision precision) {
    int64_t per = nanosecondsPer(precision);
    int64_t value = dur / per;
    dur = dur % per;
    return value;
  };

  auto stage = [&](Precision precision) -> bool {
    auto val = extract(precision);
    if (val > 0 || !result.empty())
      result += fmt::format("{}{}", val, precisionSuffix(precision));
    if (dur == 0 || precision == max_precision) {
      if (isNegative)
        result = "-" + result;
      return true;
    }
    return false;
  };

  if (stage(Precision::Years))
    return result;

  if (stage(Precision::Months))
    return result;

  if (stage(Precision::Days))
    return result;

  if (stage(Precision::Hours))
    return result;

  if (stage(Precision::Minutes))
    return result;

  if (stage(Precision::Seconds))
    return result;

  if (stage(Precision::Milliseconds))
    return result;

  if (stage(Precision::Microseconds))
    return result;

  if (stage(Precision::Nanoseconds))
    return result;

  return result;
}

} // namespace M::motr::Time

#endif // MOTR_TIME_H
