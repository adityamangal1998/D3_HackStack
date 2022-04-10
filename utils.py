"""Utilities for logging."""
import os
import cv2
import logging
import time

ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
if os.getenv('LOG_LEVEL') == 'DEBUG':
    level = logging.DEBUG
elif os.getenv('LOG_LEVEL') == 'INFO':
    level = logging.INFO
elif os.getenv('LOG_LEVEL') == 'ERROR':
    level = logging.ERROR
else:
    level = logging.ERROR
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
logger = logging.getLogger(__name__)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger = logging.getLogger(method.__name__)
        logger.debug('{} {:.3f} sec'.format(method.__name__, te-ts))
        return result
    return timed

def debug(text,debug_bool):
    if debug_bool:
        print(f"DEBUG : {text}")