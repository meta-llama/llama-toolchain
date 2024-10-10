# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import logging
import queue
import threading
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List


from llama_stack.apis.telemetry import *  # noqa: F403


def generate_short_uuid(len: int = 12):
    full_uuid = uuid.uuid4()
    uuid_bytes = full_uuid.bytes
    encoded = base64.urlsafe_b64encode(uuid_bytes)
    return encoded.rstrip(b"=").decode("ascii")[:len]


CURRENT_TRACE_CONTEXT = None
BACKGROUND_LOGGER = None


class BackgroundLogger:
    def __init__(self, api: Telemetry, capacity: int = 1000):
        self.api = api
        self.log_queue = queue.Queue(maxsize=capacity)
        self.worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.worker_thread.start()

    def log_event(self, event):
        try:
            self.log_queue.put_nowait(event)
        except queue.Full:
            print("Log queue is full, dropping event")

    def _process_logs(self):
        while True:
            try:
                event = self.log_queue.get()
                # figure out how to use a thread's native loop
                asyncio.run(self.api.log_event(event))
            except Exception:
                import traceback

                traceback.print_exc()
                print("Error processing log event")
            finally:
                self.log_queue.task_done()

    def __del__(self):
        self.log_queue.join()


class TraceContext:
    spans: List[Span] = []

    def __init__(self, logger: BackgroundLogger, trace_id: str):
        self.logger = logger
        self.trace_id = trace_id

    def push_span(self, name: str, attributes: Dict[str, Any] = None):
        current_span = self.get_current_span()
        span = Span(
            span_id=generate_short_uuid(),
            trace_id=self.trace_id,
            name=name,
            start_time=datetime.now(),
            parent_span_id=current_span.span_id if current_span else None,
            attributes=attributes,
        )

        self.logger.log_event(
            StructuredLogEvent(
                trace_id=span.trace_id,
                span_id=span.span_id,
                timestamp=span.start_time,
                attributes=span.attributes,
                payload=SpanStartPayload(
                    name=span.name,
                    parent_span_id=span.parent_span_id,
                ),
            )
        )

        self.spans.append(span)

    def pop_span(self, status: SpanStatus = SpanStatus.OK):
        span = self.spans.pop()
        if span is not None:
            self.logger.log_event(
                StructuredLogEvent(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    timestamp=span.start_time,
                    attributes=span.attributes,
                    payload=SpanEndPayload(
                        status=status,
                    ),
                )
            )

    def get_current_span(self):
        return self.spans[-1] if self.spans else None


def setup_logger(api: Telemetry, level: int = logging.INFO):
    global BACKGROUND_LOGGER

    BACKGROUND_LOGGER = BackgroundLogger(api)
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(TelemetryHandler())


async def start_trace(name: str, attributes: Dict[str, Any] = None):
    global CURRENT_TRACE_CONTEXT, BACKGROUND_LOGGER

    if BACKGROUND_LOGGER is None:
        print("No Telemetry implementation set. Skipping trace initialization...")
        return

    trace_id = generate_short_uuid()
    context = TraceContext(BACKGROUND_LOGGER, trace_id)
    context.push_span(name, {"__root__": True, **(attributes or {})})

    CURRENT_TRACE_CONTEXT = context


async def end_trace(status: SpanStatus = SpanStatus.OK):
    global CURRENT_TRACE_CONTEXT

    context = CURRENT_TRACE_CONTEXT
    if context is None:
        return

    context.pop_span(status)
    CURRENT_TRACE_CONTEXT = None


def severity(levelname: str) -> LogSeverity:
    if levelname == "DEBUG":
        return LogSeverity.DEBUG
    elif levelname == "INFO":
        return LogSeverity.INFO
    elif levelname == "WARNING":
        return LogSeverity.WARN
    elif levelname == "ERROR":
        return LogSeverity.ERROR
    elif levelname == "CRITICAL":
        return LogSeverity.CRITICAL
    else:
        raise ValueError(f"Unknown log level: {levelname}")


# TODO: ideally, the actual emitting should be done inside a separate daemon
# process completely isolated from the server
class TelemetryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        # horrendous hack to avoid logging from asyncio and getting into an infinite loop
        if record.module in ("asyncio", "selector_events"):
            return

        global CURRENT_TRACE_CONTEXT, BACKGROUND_LOGGER

        if BACKGROUND_LOGGER is None:
            raise RuntimeError("Telemetry API not initialized")

        context = CURRENT_TRACE_CONTEXT
        if context is None:
            return

        span = context.get_current_span()
        if span is None:
            return

        BACKGROUND_LOGGER.log_event(
            UnstructuredLogEvent(
                trace_id=span.trace_id,
                span_id=span.span_id,
                timestamp=datetime.now(),
                message=self.format(record),
                severity=severity(record.levelname),
            )
        )

    def close(self):
        pass


class SpanContextManager:
    def __init__(self, name: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.attributes = attributes

    def __enter__(self):
        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT
        if context:
            context.push_span(self.name, self.attributes)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT
        if context:
            context.pop_span()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.__exit__(exc_type, exc_value, traceback)

    def __call__(self, func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)
            else:
                return sync_wrapper(*args, **kwargs)

        return wrapper


def span(name: str, attributes: Dict[str, Any] = None):
    return SpanContextManager(name, attributes)
