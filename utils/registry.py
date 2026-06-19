from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry(dict[str, T]):
    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            self[name] = obj
            return obj

        return decorator
