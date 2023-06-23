from typing import Literal, TypedDict, Any


class DraughtsConfig(TypedDict, total=False):
    mustForceTake: bool
    canTakeBackwards: bool
    canChainTake: bool
    field: int
    hasFlyingKings: bool
    isPromotedImmediately: bool
    mustTakeLongestChain: bool
    rows: int
    staleThreshold: int
    stuckThreshold: int
    threefoldRepetition: bool
    whitePlaysFirst: bool
    winningCondition: Literal["elimination", "otherside"]
    clock: int
