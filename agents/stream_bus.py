"""Distributed & In-Memory SSE Stream Event Bus for genAI.

Decouples the FastAPI streaming generator from the autonomous agent state machine:
1. In-Memory Ring Buffer with Last-Event-ID Replay:
   - Caches events per stream/session so dropped connections can reconnect mid-stream.
   - Automatically replays any events missed during network disconnections.
   - Bounded ring buffer (max 1000 events) with TTL-based expiration (30 minutes).
2. Background Execution Decoupling:
   - Agent execution continues safely in the background even if the client socket drops.
   - Persists state and reports to SQLite WAL on completion regardless of client connectivity.
3. Optional Redis Pub/Sub:
   - If REDIS_URL is configured and redis-py is available, events are broadcasted to Redis.
   - Seamlessly falls back to in-memory mode for zero-dependency standalone execution.
"""

import os
import re
import json
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class StreamEvent:
    """Individual Server-Sent Event with metadata."""
    event_id: int
    raw: str
    event_type: str = "message"
    timestamp: float = field(default_factory=time.time)


class StreamChannel:
    """Channel representing an active or completed streaming session with event replay buffer."""

    def __init__(self, stream_id: str, session_id: Optional[str] = None, max_buffer_size: int = 1000):
        self.stream_id = stream_id
        self.session_id = session_id
        self.max_buffer_size = max_buffer_size
        self.events: List[StreamEvent] = []
        self.is_running = True
        self.is_completed = False
        self.is_cancelled = False
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.next_event_id = 1
        
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = threading.RLock()

    def _extract_event_id(self, raw_event: str) -> int:
        """Extract event ID from 'id: <number>' in SSE chunk or allocate next sequence."""
        match = re.search(r"^id:\s*(\d+)", raw_event, re.MULTILINE)
        if match:
            return int(match.group(1))
        current_id = self.next_event_id
        self.next_event_id += 1
        return current_id

    def _extract_event_type(self, raw_event: str) -> str:
        """Extract event type from SSE payload if available."""
        try:
            match = re.search(r"data:\s*(\{.*\})", raw_event, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1).strip())
                return parsed.get("type", "message")
        except Exception:
            pass
        return "message"

    def publish(self, raw_event: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> StreamEvent:
        """Record an SSE event into the buffer and broadcast to all connected subscriber queues."""
        with self._lock:
            eid = self._extract_event_id(raw_event)
            etype = self._extract_event_type(raw_event)
            event_obj = StreamEvent(event_id=eid, raw=raw_event, event_type=etype, timestamp=time.time())

            # Maintain ring buffer
            self.events.append(event_obj)
            if len(self.events) > self.max_buffer_size:
                self.events.pop(0)

            self.updated_at = time.time()
            if eid >= self.next_event_id:
                self.next_event_id = eid + 1

            # Dispatch to async subscriber queues
            subscribers_snapshot = list(self._subscribers)

        for q in subscribers_snapshot:
            if loop and loop.is_running():
                try:
                    loop.call_soon_threadsafe(q.put_nowait, event_obj)
                except Exception:
                    pass
            else:
                try:
                    q.put_nowait(event_obj)
                except Exception:
                    pass

        return event_obj

    def get_events_since(self, last_event_id: int) -> List[StreamEvent]:
        """Retrieve all buffered events with event_id strictly greater than last_event_id."""
        with self._lock:
            return [ev for ev in self.events if ev.event_id > last_event_id]

    def subscribe(self, queue: asyncio.Queue) -> None:
        """Register an active async consumer queue."""
        with self._lock:
            self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a disconnected async consumer queue."""
        with self._lock:
            self._subscribers.discard(queue)

    def complete(self, error: Optional[str] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Mark stream completed and send completion sentinel (None) to all subscribers."""
        with self._lock:
            self.is_running = False
            self.is_completed = True
            self.error = error
            self.updated_at = time.time()
            subscribers_snapshot = list(self._subscribers)

        for q in subscribers_snapshot:
            if loop and loop.is_running():
                try:
                    loop.call_soon_threadsafe(q.put_nowait, None)
                except Exception:
                    pass
            else:
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

    def cancel(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Cancel execution of the active channel."""
        with self._lock:
            self.is_cancelled = True
            self.is_running = False
            self.updated_at = time.time()
            subscribers_snapshot = list(self._subscribers)

        for q in subscribers_snapshot:
            if loop and loop.is_running():
                try:
                    loop.call_soon_threadsafe(q.put_nowait, None)
                except Exception:
                    pass
            else:
                try:
                    q.put_nowait(None)
                except Exception:
                    pass


class StreamEventBus:
    """Central event bus managing streams, session mappings, and reconnection replay buffers."""

    def __init__(self):
        self._channels: Dict[str, StreamChannel] = {}
        self._session_to_stream: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._redis_client = None
        self._init_redis()

    def _init_redis(self):
        """Optional Redis connection for distributed pub/sub."""
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                print(f"[StreamBus] Connected to Redis pub/sub at {redis_url}")
            except Exception as e:
                print(f"[StreamBus] Redis connection failed ({e}). Defaulting to standalone in-memory mode.")
                self._redis_client = None

    def create_channel(self, stream_id: str, session_id: Optional[str] = None) -> StreamChannel:
        """Create a new streaming channel."""
        with self._lock:
            channel = StreamChannel(stream_id=stream_id, session_id=session_id)
            self._channels[stream_id] = channel
            if session_id:
                self._session_to_stream[session_id] = stream_id
            self.cleanup_expired()
            return channel

    def get_channel(self, stream_id: str) -> Optional[StreamChannel]:
        """Fetch an existing stream channel by ID."""
        with self._lock:
            return self._channels.get(stream_id)

    def get_active_stream_for_session(self, session_id: str) -> Optional[str]:
        """Retrieve stream_id mapped to session_id if exists and active."""
        with self._lock:
            stream_id = self._session_to_stream.get(session_id)
            if stream_id and stream_id in self._channels:
                return stream_id
            return None

    def is_channel_running(self, stream_id: str) -> bool:
        """Check whether a channel is currently running."""
        channel = self.get_channel(stream_id)
        return channel.is_running if channel else False

    def publish(self, stream_id: str, raw_event: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[StreamEvent]:
        """Publish an event to a channel and broadcast via in-memory and optional Redis."""
        channel = self.get_channel(stream_id)
        if not channel:
            return None

        event = channel.publish(raw_event, loop=loop)

        # Optional Redis broadcast
        if self._redis_client:
            try:
                self._redis_client.publish(f"stream:{stream_id}", raw_event)
            except Exception:
                pass

        return event

    def finish_stream(self, stream_id: str, error: Optional[str] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Mark a stream as finished."""
        channel = self.get_channel(stream_id)
        if channel:
            channel.complete(error=error, loop=loop)

    def cancel_stream(self, stream_id: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Cancel a stream."""
        channel = self.get_channel(stream_id)
        if channel:
            channel.cancel(loop=loop)

    def cleanup_expired(self, ttl_seconds: int = 1800) -> int:
        """Prune channels that have been completed and inactive for more than ttl_seconds (default 30 mins)."""
        now = time.time()
        to_delete = []
        with self._lock:
            for sid, ch in self._channels.items():
                if ch.is_completed and (now - ch.updated_at) > ttl_seconds:
                    to_delete.append(sid)

            for sid in to_delete:
                ch = self._channels.pop(sid, None)
                if ch and ch.session_id and self._session_to_stream.get(ch.session_id) == sid:
                    self._session_to_stream.pop(ch.session_id, None)

        return len(to_delete)


# Global singleton instance
stream_bus = StreamEventBus()
