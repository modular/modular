# Logging library

Use this library to emit log messages from any layer of the stack to a file or
stdout, with timestamps and severity levels.

## Macros

`MLOG_DEBUG`, `MLOG_INFO`, `MLOG_WARN`, `MLOG_ERROR`, and `MLOG_FATAL`
are convenience wrappers around `MLOG(level, "format string", args...)`, which
emits a message at the specified level to a file or stdout. The format string
uses `fmt` syntax, which is similar to `std::format`.

## Environment variables

The following environment variables control logging behavior.

| Variable                   | Description                                                           |
|----------------------------|-----------------------------------------------------------------------|
| `MODULAR_LOG_FILE`         | Path to the log file. If unset, output goes to stdout.                |
| `MODULAR_LOG_ISO_TIME`     | Output timestamps in `YYYY-MM-DD:hh:mm:ss` format.                    |
| `MODULAR_LOG_LEVEL`        | Minimum message level to write. Corresponds to the macro names above. |
| `MODULAR_LOG_MICROSECONDS` | Include microseconds in the timestamp.                                |
| `MODULAR_LOG_NO_ENHANCED`  | Disable all prefix formatting, including the level and timestamp.     |
| `MODULAR_LOG_NO_TIMESTAMP` | Disable the timestamp while keeping the level prefix.                 |
| `MODULAR_LOG_JSON`         | Output JSON log lines, overriding other output configurations.        |

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
