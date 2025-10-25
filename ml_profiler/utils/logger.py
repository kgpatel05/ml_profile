import logging
import os


_LEVEL = os.getenv("ML_PROFILER_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=_LEVEL,
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
)


logger = logging.getLogger("ml_profiler")
