"""
Event Bus for FibriNet Simulation.

Simple pub/sub implementation with:
- Synchronous delivery
- Handler isolation (one handler error must not crash publish)
- No Tkinter imports
"""

import logging
from typing import Callable, Dict, List, Optional, Type, TypeVar, Any
from dataclasses import dataclass
from src.events.events import Event

logger = logging.getLogger(__name__)

# Type variable for event types
E = TypeVar('E', bound=Event)

# Handler type: function that takes an event
EventHandler = Callable[[Event], None]


@dataclass
class HandlerRegistration:
    """Registration record for an event handler."""
    handler: EventHandler
    handler_id: int
    event_type: Type[Event]
    name: Optional[str] = None

    def __init__(self, handler: EventHandler, handler_id: int, event_type: Type[Event], name: str = None):
        self.handler = handler
        self.handler_id = handler_id
        self.event_type = event_type
        self.name = name or handler.__name__ if hasattr(handler, '__name__') else f"handler_{handler_id}"


class EventBus:
    """
    Simple pub/sub event bus.

    Features:
    - Subscribe handlers to specific event types
    - Publish events to all registered handlers
    - Handler isolation: one handler error does not crash others
    - Synchronous delivery (handlers called in registration order)

    Usage:
        bus = EventBus()

        def on_batch(event: BatchCompleted):
            print(f"Batch {event.batch_index} completed")

        bus.subscribe(BatchCompleted, on_batch)
        bus.publish(BatchCompleted(batch_index=1))
    """

    def __init__(self, name: str = "default"):
        """
        Initialize event bus.

        Args:
            name: Name for logging purposes
        """
        self._name = name
        self._handlers: Dict[Type[Event], List[HandlerRegistration]] = {}
        self._next_handler_id = 0
        self._error_count = 0
        self._publish_count = 0

    @property
    def name(self) -> str:
        """Bus name."""
        return self._name

    @property
    def error_count(self) -> int:
        """Number of handler errors encountered."""
        return self._error_count

    @property
    def publish_count(self) -> int:
        """Number of events published."""
        return self._publish_count

    def subscribe(
        self,
        event_type: Type[E],
        handler: Callable[[E], None],
        name: Optional[str] = None,
    ) -> int:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Event class to subscribe to
            handler: Function to call when event is published
            name: Optional name for the handler (for debugging)

        Returns:
            Handler ID (can be used to unsubscribe)
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        handler_id = self._next_handler_id
        self._next_handler_id += 1

        registration = HandlerRegistration(
            handler=handler,
            handler_id=handler_id,
            event_type=event_type,
            name=name,
        )
        self._handlers[event_type].append(registration)

        logger.debug(
            f"EventBus[{self._name}]: Registered handler {registration.name} "
            f"for {event_type.__name__} (id={handler_id})"
        )

        return handler_id

    def unsubscribe(self, handler_id: int) -> bool:
        """
        Unsubscribe a handler by ID.

        Args:
            handler_id: ID returned by subscribe()

        Returns:
            True if handler was found and removed
        """
        for event_type, handlers in self._handlers.items():
            for i, reg in enumerate(handlers):
                if reg.handler_id == handler_id:
                    handlers.pop(i)
                    logger.debug(
                        f"EventBus[{self._name}]: Unsubscribed handler {reg.name} "
                        f"from {event_type.__name__}"
                    )
                    return True
        return False

    def unsubscribe_all(self, event_type: Optional[Type[Event]] = None) -> int:
        """
        Unsubscribe all handlers.

        Args:
            event_type: If provided, only unsubscribe handlers for this type.
                       If None, unsubscribe all handlers.

        Returns:
            Number of handlers removed
        """
        count = 0
        if event_type is not None:
            if event_type in self._handlers:
                count = len(self._handlers[event_type])
                self._handlers[event_type] = []
        else:
            for handlers in self._handlers.values():
                count += len(handlers)
            self._handlers.clear()

        logger.debug(f"EventBus[{self._name}]: Unsubscribed {count} handlers")
        return count

    def publish(self, event: Event) -> int:
        """
        Publish an event to all subscribed handlers.

        Handlers are called synchronously in registration order.
        Handler errors are logged but do not prevent other handlers from running.

        Args:
            event: Event to publish

        Returns:
            Number of handlers that executed successfully
        """
        self._publish_count += 1
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug(
                f"EventBus[{self._name}]: No handlers for {event_type.__name__}"
            )
            return 0

        success_count = 0
        for reg in handlers:
            try:
                reg.handler(event)
                success_count += 1
            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"EventBus[{self._name}]: Handler {reg.name} raised {type(e).__name__}: {e}"
                )
                # Continue to next handler - don't crash the whole publish

        logger.debug(
            f"EventBus[{self._name}]: Published {event_type.__name__} to "
            f"{success_count}/{len(handlers)} handlers"
        )

        return success_count

    def has_handlers(self, event_type: Type[Event]) -> bool:
        """
        Check if any handlers are registered for an event type.

        Args:
            event_type: Event class to check

        Returns:
            True if at least one handler is registered
        """
        return bool(self._handlers.get(event_type))

    def handler_count(self, event_type: Optional[Type[Event]] = None) -> int:
        """
        Count registered handlers.

        Args:
            event_type: If provided, count handlers for this type only.
                       If None, count all handlers.

        Returns:
            Number of handlers
        """
        if event_type is not None:
            return len(self._handlers.get(event_type, []))
        return sum(len(handlers) for handlers in self._handlers.values())

    def get_handler_names(self, event_type: Type[Event]) -> List[str]:
        """
        Get names of all handlers for an event type.

        Args:
            event_type: Event class

        Returns:
            List of handler names
        """
        handlers = self._handlers.get(event_type, [])
        return [reg.name for reg in handlers]

    def clear_stats(self) -> None:
        """Reset error and publish counters."""
        self._error_count = 0
        self._publish_count = 0

    def __repr__(self) -> str:
        total = self.handler_count()
        return f"EventBus(name={self._name!r}, handlers={total})"
