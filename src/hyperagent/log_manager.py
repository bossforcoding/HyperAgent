from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import datetime
from typing import Any, cast

# ===== Global state for idempotency / locking =====
_LOCK = threading.RLock()
_RUN_ID: str | None = None


def _run_id() -> str:
    """Stable ID per process run (timestamp + PID), resilient to fork."""
    global _RUN_ID
    current_pid = os.getpid()
    # If cached _RUN_ID belongs to a different PID (after fork), drop it.
    if _RUN_ID and not _RUN_ID.endswith(f"_{current_pid}"):
        _RUN_ID = None
    if _RUN_ID is None:
        _RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{current_pid}"
    return _RUN_ID


# ===== Typed custom Logger to carry internal flags =====
class HyperLogger(logging.Logger):
    """Custom logger that carries internal state for the logging setup."""
    _hyperagent_configured: bool
    _hyperagent_run_id: str

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self._hyperagent_configured = False
        self._hyperagent_run_id = ""


# Ensure future getLogger() calls create HyperLogger instances
logging.setLoggerClass(HyperLogger)


class _ExcludeLoggerPrefixFilter(logging.Filter):
    """Filter-out records whose logger name starts with any of the given prefixes."""

    def __init__(self, *prefixes: str) -> None:
        super().__init__()
        self._prefixes = tuple(p for p in prefixes if p)

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return not any(name == p or name.startswith(p + ".") for p in self._prefixes)


# ===== Colored console formatter =====
class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on log level for TTY consoles."""

    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # BASE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    BASE = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    FORMATS = {
        logging.DEBUG: GREY + BASE + RESET,
        logging.INFO: GREY + BASE + RESET,
        logging.WARNING: YELLOW + BASE + RESET,
        logging.ERROR: RED + BASE + RESET,
        logging.CRITICAL: BOLD_RED + BASE + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.GREY + self.BASE + self.RESET)
        return logging.Formatter(log_fmt).format(record)


# ===== Plain file formatter (no ANSI) =====
_FILE_FORMATTER = logging.Formatter(
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


def _normalize_level(level) -> int:
    """Return a numeric logging level from int or str; fallback to DEBUG."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        s = level.strip()
        if s.isdigit():
            return int(s)
        n = logging.getLevelName(s.upper())
        return n if isinstance(n, int) else logging.DEBUG
    return logging.DEBUG


def _resolve_logger_name(name: str | None) -> str:
    """Resolve logger name with precedence: explicit arg > env > default."""
    env_name = os.environ.get("HYPERAGENT_LOGGER_NAME")
    if isinstance(name, str):
        stripped = name.strip()
        if stripped:
            return stripped
    if env_name:
        return env_name
    return "HyperAgent"


def get_hlogger(name: str | None = None) -> HyperLogger:
    """Typed accessor that returns a HyperLogger."""
    logger_name = _resolve_logger_name(name)
    return cast(HyperLogger, logging.getLogger(logger_name))


def setup_logger(
        *,
        name: str | None = None,
        console_level: str | int | None = None,
        file_level: str | int | None = None,
        logger_level: str | int | None = None,
        log_dir: str | None = None,
        force: bool = False,
        autogen_io: bool = True,
) -> logging.Logger:
    """
    Configure and return a process-wide shared logger with per-handler levels.

    Behavior & features:
    - Idempotent and thread-safe (single configuration per process).
    - One log file per process run: <name>_YYYYmmdd_HHMMSS_<PID>.log
    - Colored console output (TTY only; disabled if NO_COLOR is set).
    - Third-party noise reduction (e.g., httpx, multilspy).
    - Optional pyautogen integration via LoggingIOConsole (extends IOConsole).

    Levels:
    - `console_level` controls console verbosity (default: INFO).
    - `file_level` controls file verbosity (default: DEBUG).
    - `logger_level` controls the parent logger threshold; if not provided,
       it defaults to `min(console_level, file_level)` to ensure no handler is starved.

    Environment variables (used if args are not provided):
      - HYPERAGENT_LOGGER_NAME    → logger name (default: "HyperAgent")
      - HYPERAGENT_CONSOLE_LEVEL  → console level (default: "INFO")
      - HYPERAGENT_FILE_LEVEL     → file level (default: "DEBUG")
      - HYPERAGENT_LOGGER_LEVEL   → parent logger level (optional; overrides the default)
      - HYPERAGENT_LOG_DIR        → directory for log files (default: "./logs")

    Args:
        name: Logger name. If None, taken from env or "HyperAgent".
        console_level: Console threshold (e.g., "INFO", 20). Default: INFO.
        file_level: File threshold (e.g., "DEBUG", 10). Default: DEBUG.
        logger_level: Parent logger threshold; if None, uses min(console_level, file_level).
        log_dir: Directory for the log file. If None, env or "./logs".
        force: If True, remove existing handlers and reconfigure from scratch.
        autogen_io: If True, set LoggingIOConsole as the IOStream global default (registers LoggingIOConsole
                    so pyautogen output is routed through this logger).

    Returns:
        The configured `logging.Logger`.
    """
    with _LOCK:
        # --- Resolve configuration (args > env > defaults) ---
        logger = get_hlogger(name)
        logger_name = logger.name
        console_level = _normalize_level(
            console_level or os.environ.get("HYPERAGENT_CONSOLE_LEVEL", "INFO")
        )
        file_level = _normalize_level(
            file_level or os.environ.get("HYPERAGENT_FILE_LEVEL", "DEBUG")
        )

        # Logger level: explicit override wins; else min(console, file)
        if logger_level is not None or os.environ.get("HYPERAGENT_LOGGER_LEVEL"):
            logger_level = _normalize_level(
                logger_level or os.environ["HYPERAGENT_LOGGER_LEVEL"]
            )
        else:
            logger_level = min(console_level, file_level)

        log_dir = log_dir or os.environ.get("HYPERAGENT_LOG_DIR", os.path.join(os.getcwd(), "logs"))

        run_id = _run_id()

        # If already configured and not forced, return as-is
        if getattr(logger, "_hyperagent_configured", False) and not force:
            return logger

        # If force, clean up existing handlers first
        if force:
            for h in list(logger.handlers):
                try:
                    h.close()
                finally:
                    logger.removeHandler(h)

        logger.setLevel(logger_level)
        # Do not propagate to root logger to avoid double emission if root has handlers
        logger.propagate = False

        # --- Console handler ---
        console = logging.StreamHandler()
        console.setLevel(console_level)

        use_color = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
        console.setFormatter(_ColorFormatter() if use_color else _FILE_FORMATTER)

        # Exclude LoggingIOConsole logs from console, but still allow them to propagate to file
        console.addFilter(_ExcludeLoggerPrefixFilter(f"{logger.name}.LoggingIOConsole"))

        setattr(console, "_hyperagent_run_id", run_id)
        logger.addHandler(console)

        # --- File handler (created once per run) ---
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            # Fall back to CWD if directory creation fails
            log_dir = os.getcwd()

        file_name = f"{logger_name.lower()}_{run_id}.log"
        file_path = os.path.join(log_dir, file_name)

        # delay=True defers file opening until first emit
        file_handler = logging.FileHandler(file_path, encoding="utf-8", delay=True)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(_FILE_FORMATTER)
        setattr(file_handler, "_hyperagent_run_id", run_id)
        logger.addHandler(file_handler)

        # --- Reduce verbosity from noisy libraries ---
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("multilspy").setLevel(logging.FATAL)

        # --- Optional integration with pyautogen IO ---
        if autogen_io:
            try:
                from autogen.io import IOStream, IOConsole
            except Exception:
                # autogen not installed or incompatible, ignore silently
                pass
            else:
                class LoggingIOConsole(IOConsole):
                    """
                    IOConsole implementation that prints to the console and also logs each message to a child logger "<name>.LoggingIOConsole" at INFO level.

                    Notes:
                    - Messages propagate to the parent logger, so they end up in the file handler.
                    - Use a console handler filter to hide these logs from the console if desired.
                    """

                    def print(
                            self,
                            *objects: Any,
                            sep: str = " ",
                            end: str = "\n",
                            flush: bool = False,
                    ) -> None:
                        msg = sep.join(map(str, objects))
                        io_logger = get_hlogger(f"{logger_name}.LoggingIOConsole")
                        # Ensure child logger emits at least at INFO and propagates to parent
                        if io_logger.level == logging.NOTSET:
                            io_logger.setLevel(logging.INFO)
                        io_logger.propagate = True
                        io_logger.info(msg)
                        super().print(*objects, sep=sep, end=end, flush=flush)

                # Set the default input/output stream to the console and logs
                IOStream.set_global_default(LoggingIOConsole())
                IOStream.set_default(LoggingIOConsole())
                # Allow independent tuning of LoggingIOConsole messages if needed
                get_hlogger(f"{logger_name}.LoggingIOConsole").setLevel(logging.INFO)

        # Mark as configured to keep idempotency across repeated calls
        logger._hyperagent_configured = True
        logger._hyperagent_run_id = run_id

        return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Convenience accessor that returns the configured logger,
    configuring it on first use.
    """
    return setup_logger(name=name)
