import logging
import random
import time

logger = logging.getLogger("evaluator")


def retry_on_rate_limit(fn, *args, max_attempts=6, **kwargs):
    """Call fn, retrying with exponential backoff on HF hub 429s.

    Many parallel runs share the HF API quota (1000 requests / 5 min window);
    the datasets/lm_eval task-loading paths raise on 429 without retrying.
    """
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429 or attempt == max_attempts - 1:
                raise
            # quota window is 300s; jittered so parallel runs don't retry in sync
            delay = min(300, 30 * 2**attempt) * (1 + random.random())
            logger.warning(f"HF hub rate limit (429), retrying in {delay:.0f}s")
            time.sleep(delay)
