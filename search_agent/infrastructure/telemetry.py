from __future__ import annotations

import os
from threading import Lock

import logfire

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
        console = {"min_log_level": "info"} if settings.logfire_console else False
        logfire.configure(
            local=settings.logfire_local,
            send_to_logfire=settings.resolved_send_to_logfire(),
            token=settings.logfire_token,
            service_name=settings.logfire_service_name,
            environment=settings.logfire_environment,
            console=console,
        )
        try:
            logfire.instrument_requests()
        except RuntimeError:
            pass
        logfire.instrument_pydantic()
        _CONFIGURED = True
