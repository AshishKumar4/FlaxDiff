import logging
import colorlog

logger = logging.getLogger("dataprocessor")

handler = logging.StreamHandler()

# This formatter prints:
#   - A colored log level
#   - The time
#   - The level
#   - The full path with line number (pathname:lineno)
#   - The function name
#   - The actual log message
#
# Using `%(pathname)s:%(lineno)d` is typically clickable in VSCode if the file
# path matches your local project structure.
formatter = colorlog.ColoredFormatter(
    fmt="%(log_color)s%(asctime)s [%(levelname)s] %(pathname)s:%(lineno)d in %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    },
    style="%"  # new style
)

handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(logging.INFO)  # or INFO, if you prefer less verbosity
logger.propagate = False
logging.getLogger().setLevel(logging.INFO)