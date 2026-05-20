# Logging library

Use this library to emit log messages from any layer of the stack to a file or
stdout, with timestamps and severity levels.

## Interface

The logger generally expects a single formatting string and a list of arguments
to be formatted within it. The format string uses `fmt` syntax, which is
similar to `std::format`.

### C++

`MLOG_DEBUG`, `MLOG_INFO`, `MLOG_WARN`, `MLOG_ERROR`, and `MLOG_FATAL`
are convenience wrappers around `MLOG(level, "format string", args...)`, which
emits a message at the specified level to a file or stdout. `MLOG_FATAL` will
abort the user program after logging the message.

### Mojo

There is a Mojo interface wrapping the C++ Log library. It uses the same `fmt`
formatting underneath, so the same log message will produce the same output
regardless of source (modulo timestamps etc.). The interface specifically is:

```mojo
mlog[MLOG_INFO, "format string here: {}"]("arguments here")
mlog_info["all {} log convenience functions work"](5)
```

Arguments are captured and transformed into a form suitable for the FFI call,
which is the LogArg class on the C++ side of the implementation.

## Environment variables

The following environment variables control logging behavior.

| Variable                   | Description                                                           |
|----------------------------|-----------------------------------------------------------------------|
| `MODULAR_LOG_STDOUT`       | `false` suppresses stdout output. Default true. See note on sinks.    |
| `MODULAR_LOG_FILE`         | Path to the log file. If unset, no file is written.                   |
| `MODULAR_LOG_ISO_TIME`     | Output timestamps in `YYYY-MM-DD:hh:mm:ss` format.                    |
| `MODULAR_LOG_LEVEL`        | Minimum message level to write. Corresponds to the macro names above. |
| `MODULAR_LOG_MICROSECONDS` | Include microseconds in the timestamp.                                |
| `MODULAR_LOG_NO_ENHANCED`  | Disable all prefix formatting, including the level and timestamp.     |
| `MODULAR_LOG_NO_TIMESTAMP` | Disable the timestamp while keeping the level prefix.                 |
| `MODULAR_LOG_JSON`         | Output JSON log lines, overriding other output configurations.        |

## Output sinks

Output can be sent to stdout (`MODULAR_LOG_STDOUT` is true) or to a file
(`MODULAR_LOG_FILE` is set to some valid path). These options are orthogonal;
if both are set, output goes to both, and if set to `false` and `""` (empty
string or unset) then the logging is effectively turned off.

## JSON output format

When `MODULAR_LOG_JSON` is set, each log line is a self-contained JSON object
followed by a newline (newline-delimited JSON / NDJSON). Other formatting flags
(`MODULAR_LOG_ISO_TIME`, `MODULAR_LOG_NO_TIMESTAMP`, etc.) are ignored in this
mode.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["timestamp", "level", "message"],
  "additionalProperties": false,
  "properties": {
    "timestamp": {
      "type": "string",
      "description": "UTC time in ISO 8601 format with microsecond precision.",
      "examples": ["2026-03-16T12:00:00.123456Z"]
    },
    "level": {
      "type": "string",
      "enum": ["DBG", "INFO", "WARN", "ERR", "FATL"],
      "description": "Severity level of the log message."
    },
    "message": {
      "type": "string",
      "description": "Log message text."
    }
  }
}
```
