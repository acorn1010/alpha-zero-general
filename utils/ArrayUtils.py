from typing import TypeVar


T = TypeVar('T')


def deep_unique(values: list[T]) -> list[T]:
    """Returns a list of unique values from a list"""
    return [v for i, v in enumerate(values) if v not in values[:i]]


def flatten(values: list[list[T]] | list[T]) -> list[T]:
    """Flattens a list of lists"""
    return [v for sublist in values for v in sublist]
