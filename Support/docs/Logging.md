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
