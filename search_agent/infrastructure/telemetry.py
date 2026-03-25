from __future__ import annotations

import os
from threading import Lock

import logfire

from search_agent import tuning
from search_agent.settings import AppSettings

_LOCK = Lock()
_CONFIGURED = False


def configure_logfire(settings: AppSettings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    with _LOCK:
        if _CONFIGURED:
            return

        os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
        console = {"min_log_level": "info"} if tuning.LOGFIRE_CONSOLE else False
        logfire.configure(
            local=tuning.LOGFIRE_LOCAL,
            send_to_logfire=tuning.LOGFIRE_SEND_TO_LOGFIRE,
            token=settings.logfire_token,
            service_name=tuning.LOGFIRE_SERVICE_NAME,
            environment=tuning.LOGFIRE_ENVIRONMENT,
            console=console,
        )
        try:
            logfire.instrument_requests()
        except RuntimeError:
            pass
        logfire.instrument_pydantic()
        _CONFIGURED = True
