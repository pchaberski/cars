"""
Execution time measurement utility.

Usage:
- decorate function defintion with `@timer` and allow for **kwargs
- provide `get_time=True` keyword argument to measure and print execution time
- provide valid logger in `logger` kwarg to log time instead of printing
"""

import logging
import time


def timer(function):
    """Measure execution time."""

    def wrapper(*args, **kwargs):

        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()

        if 'get_time' in kwargs:
            if kwargs['get_time']:
                time_diff = (t1 - t0) * 1000.
                time_msg = f'Execution time ({function.__name__}): {time_diff:.3f} ms'
                if 'logger' in kwargs:
                    logger = kwargs['logger']
                    if isinstance(logger, logging.Logger):
                        logger.debug(time_msg)
                else:
                    print(time_msg)

        return result

    return wrapper
