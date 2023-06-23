from draughts.DraughtsConfig import DraughtsConfig
from draughts.DraughtsLogic import Board


SHARED_CONFIG = {
    'clock': 60_000,
    'winningCondition': 'elimination',
    'staleThreshold': 100,
    'stuckThreshold': 1,
}

INTERNATIONAL_CONFIG = {
    **SHARED_CONFIG,
    'mustForceTake': True,
    'canTakeBackwards': True,
    'canChainTake': True,
    'field': 10,
    'hasFlyingKings': True,
    'isPromotedImmediately': False,
    'mustTakeLongestChain': True,
    'rows': 4,
    'threefoldRepetition': True,
    'whitePlaysFirst': True,
}

CONFIGS = {
    'checkers': DraughtsConfig(
        **SHARED_CONFIG,
        mustForceTake=True,
        canTakeBackwards=False,
        canChainTake=True,
        field=8,
        hasFlyingKings=False,
        isPromotedImmediately=True,
        mustTakeLongestChain=False,
        rows=3,
        threefoldRepetition=False,
        whitePlaysFirst=False,
    ),
    'international': DraughtsConfig(**INTERNATIONAL_CONFIG),
    'russian': DraughtsConfig(**INTERNATIONAL_CONFIG),
    'brazilian': DraughtsConfig(**INTERNATIONAL_CONFIG),
    'canadian': DraughtsConfig(**INTERNATIONAL_CONFIG),
}
CONFIGS['russian']['field'] = 8
CONFIGS['russian']['isPromotedImmediately'] = True
CONFIGS['russian']['mustTakeLongestChain'] = False
CONFIGS['russian']['rows'] = 3

CONFIGS['brazilian']['field'] = 8
CONFIGS['brazilian']['rows'] = 3

CONFIGS['canadian']['field'] = 12
CONFIGS['canadian']['rows'] = 5


class TestDraughtsLogic:
    def test_black_must_jump(self):
        engine = make_engine('1. 12-16 24-19 2. 16-20 19-16 3. 8-12 22-17', CONFIGS['checkers'])
        assert len(engine.moves()) == 1
        assert move(engine, 'h6', 'f4') is not None

    def test_black_can_double_capture_backwards(self):
        engine = make_engine('pdn 1. 23-18 2. 9-13 22-17', CONFIGS['russian'])
        assert move(engine, 'a5', 'e5') is not None

    def test_flying_kings_can_move_any_number_of_spaces(self):
        engine = make_engine('pdn 1. g3-h4 2. d6-c5 c3-b4 3. c5-d4 e3xc5 4. b6xd4 d2-c3 5. c7-d6 c3xc7 6. b8xd6 b4-a5 7. f6-e5 f2-e3 8. e7-f6 e3-d4 9. e5xc3 b2xd4 10. a7-b6 a5xe5 11. d8-e7 e5-d6 12. e7xe3 e1-d2 13. f6-g5 h4xf6 14. g7xe5 d2xd6 15. f8-e7 d6xf8 16. h8-g7', CONFIGS['russian'])
        assert move(engine, 'f8', 'd6') is not None

    def test_flying_king_can_only_turn_if_capturing_a_piece(self):
        engine = make_engine('pdn 1. g3-h4 2. b6-c5 c3-b4 3. f6-e5 b4-a5 4. c5-b4 a5xc3 5. h6-g5 h4xd4 6. c7-b6 c3-b4 7. d8-c7 b4-a5 8. g7-f6 e3-f4 9. f8-g7 h2-g3 10. b6-c5 d4xd8 11. b8-c7', CONFIGS['russian'])
        assert move(engine, 'd8', 'c5') is None
        assert move(engine, 'd8', 'b6') is not None

    def test_flying_king_can_capture_when_other_pieces_can_capture(self):
        engine = make_engine('pdn 1. g3-h4 2. b6-c5 c3-b4 3. f6-g5 h4xf6 4. g7xe5 f2-g3 5. e7-f6 g3-h4 6. a7-b6 b4-a5 7. h8-g7 d2-c3 8. f8-e7 c3-b4 9. b8-a7 h2-g3 10. e5-f4 g3xe5 11. d6xd2 b4xf8 12. b6-c5', CONFIGS['russian'])
        assert move(engine, 'f8', 'b4') is not None

    def test_can_move_non_king_when_player_has_flying_king(self):
        engine = make_engine('pdn 1. g3-h4 2. b6-c5 c3-b4 3. f6-g5 h4xf6 4. g7xe5 f2-g3 5. e7-f6 g3-h4 6. a7-b6 b4-a5 7. h8-g7 d2-c3 8. f8-e7 c3-b4 9. b8-a7 h2-g3 10. e5-f4 g3xe5 11. d6xd2 b4xf8 12. b6-c5', CONFIGS['russian'])
        assert move(engine, 'a7', 'b6') is not None

    def test_king_can_capture_backwards_when_not_flying_king(self):
        engine = make_engine('pdn 1. 32-28 17-22 2. 28x17 11x22 3. 31-27 22x31 4. 36x27 19-23 5. 27-21 16x27 6. 34-30 20-24 7. 30x19 14x23 8. 33-28 10-14 9. 35-30 14-20 10. 30-25 23-29 11. 25-20 15x24 12. 38-33 29x40 13. 39-34 40x29 14. 33x35 9-14 15. 44-39 24-30 16. 35x24 14-20 17. 24x15 13-19 18. 15x24 7-12 19. 37-32 12-17 20. 41-37 8-12 21. 37-31 17-21 22. 31-27 12-17 23. 39-33 4-10 24. 45-40 21-26 25. 33-29 26x37 26. 40-35 10-15 27. 29-24 3-9 28. 35-30 9-14 29. 24-20 15x24 30. 30x10 5x14 31. 32-28', CONFIGS['international'])
        assert move(engine, 'g1', 'e3') is not None


def move(engine: Board, from_square: str, to_square: str):
    return engine.move(engine.get_coordinate_from_square(from_square), engine.get_coordinate_from_square(to_square))


DEFAULT_CONFIG: DraughtsConfig = DraughtsConfig(**{
    'mustForceTake': True,
    'canTakeBackwards': False,
    'canChainTake': True,
    'field': 8,
    'hasFlyingKings': False,
    'isPromotedImmediately': True,
    'mustTakeLongestChain': False,
    'rows': 3,
    'staleThreshold': 100,
    'stuckThreshold': 1,
    'threefoldRepetition': False,
    'whitePlaysFirst': False,
    'winningCondition': 'elimination',
    'clock': 0,
})


def make_engine(pdn: str, config: DraughtsConfig):
    result = Board(DraughtsConfig(**{**DEFAULT_CONFIG, **(config or {})}))
    result.load_pdn(pdn)
    return result
