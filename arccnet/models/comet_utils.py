"""Lightweight shared helpers for Comet logger interactions."""


def safe_comet_call(comet_logger, logger, action: str, method_name: str, *args, **kwargs) -> bool:
    """
    Call a Comet experiment method safely so logging failures never crash training.

    Returns True when the method exists and executes successfully, else False.
    """
    if comet_logger is None:
        return False

    experiment = getattr(comet_logger, "experiment", None)
    if experiment is None:
        return False

    method = getattr(experiment, method_name, None)
    if method is None:
        return False

    try:
        method(*args, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("Comet %s failed: %s", action, exc)
        return False
