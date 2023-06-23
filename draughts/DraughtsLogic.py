import dataclasses
import re
from copy import copy
from functools import reduce
from operator import concat
from typing import Literal, NamedTuple

import numpy as np

from draughts.DraughtsConfig import DraughtsConfig
from utils.ArrayUtils import deep_unique


BOARD_COLUMNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']

PLAYER_BLACK: Literal[-1] = -1
PLAYER_WHITE: Literal[1] = 1


@dataclasses.dataclass
class Dimensions:
    rows: int
    columns: int


class MovementDelta(NamedTuple):
    delta_x: int
    delta_y: int


@dataclasses.dataclass
class PieceMovePartial:
    to: tuple[int, int]

    # The team index (0-based) of the team making this move.
    team_index: Literal[-1, 1]


@dataclasses.dataclass
class PieceMove(PieceMovePartial):
    frm: tuple[int, int]

    # The number of captures on the path from `frm` to `to`.
    captures: int


@dataclasses.dataclass
class Move:
    frm: tuple[int, int]
    to: tuple[int, int]
    shorthand: Literal['m', 'k']


@dataclasses.dataclass
class PieceMoveWithCapture(PieceMove):
    is_capture: bool


MOVEMENT_DELTAS: list[MovementDelta] = [
    MovementDelta(1, 1),
    MovementDelta(1, -1),
    MovementDelta(-1, 1),
    MovementDelta(-1, -1),
]


@dataclasses.dataclass
class TypeTeamIndex:
    """Represents a piece on the board.

    Attributes:
        team_index: The team index (0-based) of the piece.
    """
    team_index: Literal[-1, 1]
    type: Literal['m', 'k']


@dataclasses.dataclass
class Piece:
    """Represents a piece on the board.

    Attributes:
        team_index: The team index (0-based) of the piece.
    """
    x: int
    y: int
    square: str
    team_index: Literal[-1, 1]
    type: Literal['m', 'k']

    def __copy__(self):
        return Piece(self.x, self.y, self.square, self.team_index, self.type)


def _get_team_count() -> int:
    return 2


class Board:
    """Contains the board state for a game of draughts.

    Attributes:
        _current_team_index: 0-based player index whose turn it currently is.
        _game_end_state: The game's end state (if the game has ended), else null. If the game has ended, this will be
            won, or 'tie' if neither team won.
        _has_capture: True if there's a capture that must be made on this turn.
        _stale_count: Number of turns (not rounds) that have elapsed without a capture taking place.
        _turn: The (0-based) turn that the game is on. Increments by 1 once every player has had a move.
        _pieces: Pieces on the board. Ordered by top-left to bottom-right going left-to-right then top-to-bottom.
        _team_to_turns_stuck: Maps team index (0-based) to the number of turns they've been stuck (unable to make a
            move).
    """

    _current_team_index = 0
    _game_end_state: int | Literal['tie'] | None = None
    _has_capture = False
    _stale_count = 0
    _turn = 0
    _pieces: np.array
    _team_to_turns_stuck: dict[int, int] = {}

    # Mapping of a game state (piece positions / types / player turn) to number
    # of times that state has been observed. If the same state is observed 3
    # times in some variants, then the game ends in a draw.
    _game_state_counts: dict[str, int] = {}

    def __init__(self, config: DraughtsConfig, pieces: np.array = None):
        """Set up initial board configuration."""
        self.config = config

        if pieces is None:
            # Set pieces to default
            self._reset()
        else:
            self._pieces = pieces

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self._pieces[index]

    def _get_team_pieces(self) -> np.array:
        """Returns all pieces for the current team."""
        return self._pieces[self._pieces.team_index == self._current_team_index]

    def _reset(self):
        """Reset the board to its initial configuration."""
        # Set pieces to default
        self._pieces = np.zeros([self.config['field'], self.config['field']], dtype=Piece)

        # Set the number of stuck turns per team to 0.
        for i in range(2):
            self._team_to_turns_stuck[i] = 0

        # Black
        rows = self.config['field']
        columns = self.config['field']
        for y in range(self.config['rows']):
            for x in range(columns):
                if (x + y) % 2 == 1:
                    self._pieces[x][y] = PLAYER_BLACK

        for y in range(rows - self.config['rows'], rows):
            for x in range(columns):
                if (x + y) % 2 == 1:
                    self._pieces[x][y] = PLAYER_WHITE

    def get_valid_moves(self) -> list[PieceMove]:
        """Returns a list of legal moves from the current position. If no position is specified, then all valid moves
        are returned for the current player."""
        minimum_captures = self._get_minimum_required_captures()
        return [move for move in self._get_unfiltered_moves() if move.captures >= minimum_captures]

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        # (x,y) = move
        # assert self[x][y] == 0
        # self[x][y] = color
        pass

    def _get_longest_chain(self, frm: tuple[int, int], to: tuple[int, int]) -> list[PieceMoveWithCapture]:
        """Returns the longest chain of moves for a piece to go from `frm` to `to`."""
        maybe_piece = self._get_piece(frm[0], frm[1])
        original_shorthand = maybe_piece.type if maybe_piece else 'm'
        chains = [
            [
                {**move, 'shorthand': self._is_promotion(move, index == len(chain) - 1) and 'k' or original_shorthand}
                for index, move in enumerate(chain)
            ]
            for chain in self._get_chained_moves(frm[0], frm[1]) if chain[-1].to == to
        ]
        return max(chains, key=len, default=[])

    def move(self, frm: tuple[int, int], to: tuple[int, int]) -> Move or None:
        if self._game_end_state is not None:
            return None  # Game has ended.
        from_piece = self._get_piece(frm[0], frm[1])
        to_piece = self._get_piece(to[0], to[1])
        if not from_piece or to_piece or from_piece.team_index != self._current_team_index:
            # Unable to make a move. Either no piece is at `from`, or there's a piece at `to`.
            return None

        longest_chain = self._get_longest_chain(frm, to)
        if not len(longest_chain) or longest_chain[-1].captures < self._get_minimum_required_captures():
            # Unable to make a move. No longest chain or not enough captures.
            return None

        # Piece is jumping or moving to position.
        shorthand = from_piece.type
        did_capture = False
        captures = 0
        for i, piece_move in enumerate(longest_chain):
            is_capture = abs(piece_move.frm[0] - piece_move.to[0]) >= 2
            did_capture = did_capture or is_capture
            captures += 1 if is_capture else 0
            if is_capture:
                # Remove piece that got captured
                self._set_piece(
                    (piece_move.frm[0] + piece_move.to[0]) // 2,
                    (piece_move.frm[1] + piece_move.to[1]) // 2,
                    0,
                )
            # TODO(acorn1010): Refactor so this logic reuses _get_longest_chain?
            if self._is_promotion(piece_move, i == len(longest_chain) - 1):
                shorthand: Literal['k', 'm'] = 'k'
                self._set_piece(
                    piece_move.to[0],
                    piece_move.to[1],
                    TypeTeamIndex(type=shorthand, team_index=piece_move.team_index))
                if self.config['winningCondition'] == 'otherside':
                    self._game_end_state = self._current_team_index
            else:
                self._set_piece(
                    piece_move.to[0],
                    piece_move.to[1],
                    self._get_piece(piece_move.frm[0], piece_move.frm[1]) or 0)
            self._set_piece(piece_move.frm[0], piece_move.frm[1], 0)

        # Increment the number of stale moves (or reset them) depending on if a capture was made.
        if did_capture:
            self._stale_count = 0
        else:
            self._stale_count += 1
            if self._stale_count >= self.config['staleThreshold']:
                self._game_end_state = 'tie'
                return Move(frm=frm, to=to, shorthand=shorthand)

        # Next player's turn.
        self._next_turn()
        return Move(frm=frm, to=to, shorthand=shorthand)

    def load_pdn(self, pdn: str):
        """Loads a pdn (same format as is returned by `#pdn`)."""
        self._reset()
        for from_index, to_index in re.findall(r'([\da-n]+)[-x]([\da-n]+)', pdn):
            is_square = from_index[0] in BOARD_COLUMNS
            from_ = self.get_coordinate_from_square(from_index) if is_square \
                else self._index_to_coordinate(int(from_index))
            to = self.get_coordinate_from_square(to_index) if is_square else self._index_to_coordinate(int(to_index))
            self.move(from_, to)

    def _get_minimum_required_captures(self) -> int:
        """Returns the minimum number of captures required for a move to be valid on this turn."""
        if not self.config['mustForceTake']:
            return 0
        moves = self._get_unfiltered_moves()
        if len(moves) == 0:
            return 0
        if self.config['mustTakeLongestChain']:
            return moves[0].captures or 0
        return 1 if moves[0].captures > 0 else 0

    def moves(self) -> list[PieceMove]:
        """Returns all valid moves for the current player."""
        minimum_captures = self._get_minimum_required_captures()
        return [move for move in self._get_unfiltered_moves() if move.captures >= minimum_captures]

    def _get_unfiltered_moves(self) -> list[PieceMove]:
        indices = self._get_team_piece_indices()
        moves_for_pieces = [self._get_moves_for_piece(x, y) for x, y in indices]
        return list(reduce(concat, moves_for_pieces))

    def _get_moves_for_piece(self, x: int, y: int) -> list[PieceMove]:
        """Returns all possible valid locations that the piece at (x, y) can move to."""
        piece = self._get_piece(x, y)
        if not piece or piece.team_index != self._current_team_index:
            return []

        chained_moves: list = self._get_chained_moves(x, y)
        # noinspection PyTypeChecker
        flattened_moves: list[PieceMove] = (
            list(reduce(concat, [PieceMove(
                frm=chain[0].frm,
                to=chain[-1].to,
                captures=chain[-1].captures,
                team_index=chain[0].team_index,
            ) for chain in chained_moves if len(chain) > 0])))
        return deep_unique(flattened_moves)

    def _next_turn(self) -> None:
        """Starts the next player's turn."""
        if self._game_end_state is not None:
            return
        self._current_team_index = (self._current_team_index + 1) % _get_team_count()
        if self._current_team_index == 0:
            self._turn += 1

        # Check for threefold repetition.
        if self._has_threefold_repetition():
            self._game_end_state = 'tie'
            return

        self._has_capture = self._calculate_has_capture()
        if self._has_move():
            self._team_to_turns_stuck[self._current_team_index] = 0
            return  # Not stuck. Continue

        # No valid move. Increment this player's turns stuck.
        turns_stuck = self._team_to_turns_stuck.get(self._current_team_index, 0) + 1
        self._team_to_turns_stuck[self._current_team_index] = turns_stuck
        if turns_stuck == self.config['stuckThreshold']:
            # Check to see if we have a winner.
            if self._calculate_winning_team():
                return  # There was a winning team.

            # No winner yet. This player just got eliminated! Remove their pieces.
            self._pieces = [piece for piece in self._pieces if piece['team_index'] != self._current_team_index]

        if self._stale_count >= self.config['staleThreshold']:
            self._game_end_state = 'tie'
            return
        self._next_turn()

    def _calculate_winning_team(self) -> bool:
        """Updates the end game state and returns `true` if the game has ended."""
        remaining_teams = set(piece['team_index'] for piece in self._pieces if piece)

        # A team won!
        if len(remaining_teams) == 1:
            self._game_end_state = list(remaining_teams)[0]
            return True

        # Filter out any teams that are stuck.
        threshold = self.config['stuckThreshold'] or 0
        remaining_teams = {team for team in remaining_teams if self._team_to_turns_stuck.get(team, 0) < threshold}

        if len(remaining_teams) == 1:
            self._game_end_state = list(remaining_teams)[0]
            return True
        elif len(remaining_teams) < 1:
            self._game_end_state = 'tie'
            return True

        return False

    def _has_move(self):
        """Returns true if the current player can make a move."""
        pieces = self._get_team_pieces()
        return any(len(self._get_moves_for_piece(x, y)) > 0 for x, y in np.ndindex(pieces.shape) if pieces[x, y])

    def _get_chained_moves(self, x: int, y: int) -> list[list[PieceMoveWithCapture]]:
        piece = self._pieces[x][y]
        if not piece:
            return []
        jumped_pieces = set[Piece]()
        jumped_pieces.add(piece)
        result = self._get_chained_moves_impl(x, y, jumped_pieces, piece, False)

        # For kings, they're able to continue moving. As long as the last move isn't a capture, add an extra move that
        # excludes it.
        extra_moves: list[list[PieceMoveWithCapture]] = []
        for moves in result:
            for i in range(len(moves) - 1, 0, -1):
                if moves[i].captures - moves[i - 1].captures != 0:
                    break
                extra_moves.append(moves[:i])

        return result + extra_moves

    def _get_chained_moves_impl(
            self,
            x: int,
            y: int,
            jumped_pieces: set[Piece],
            piece: Piece,
            is_only_jump: bool,
            directions: list[MovementDelta] = MOVEMENT_DELTAS) -> list[list[PieceMoveWithCapture]]:
        """Returns all possible valid locations that the piece at (x, y) can move to, including captures."""
        # Slow code is AWESOME. -Beef; time.sleep(5);
        result: list[list[PieceMoveWithCapture]] = []
        for delta_x, delta_y in directions:
            result += self._get_moves_at_diagonal(x, y, delta_x, delta_y, jumped_pieces, is_only_jump, piece)
        return result

    def _calculate_has_capture(self):
        """Returns `true` if any piece of the current player is required to make a jump."""
        if not self.config['mustForceTake']:
            return False
        team_pieces = self._get_team_pieces()
        for x, y_col in enumerate(team_pieces):
            for y, piece in enumerate(y_col):
                jumped_pieces = set()
                moves = (
                        self._get_moves_at_diagonal(x, y, -1, -1, jumped_pieces, True)
                        + self._get_moves_at_diagonal(x, y, -1, 1, jumped_pieces, True)
                        + self._get_moves_at_diagonal(x, y, 1, -1, jumped_pieces, True)
                        + self._get_moves_at_diagonal(x, y, 1, 1, jumped_pieces, True))
                if len(moves) > 0:
                    return True
        return False

    def _can_capture(self, piece: Piece, x: int, y: int, delta_x: int, delta_y: int) -> bool:
        """Returns `true` if the piece at (x, y) can capture a piece at (x + delta_x, y + delta_y)."""
        if piece.type != 'k' or not self.config['hasFlyingKings']:
            return False
        cur_x = x + delta_x
        cur_y = y + delta_y
        to_piece = self._get_piece(cur_x, cur_y)
        while to_piece is not None:
            if to_piece:  # There's an enemy at this spot!
                # True if there's at least 1 empty spot after (piece can be captured)
                return self._get_piece(cur_x + delta_x, cur_y + delta_y) == 0
            cur_x += delta_x
            cur_y += delta_y
            to_piece = self._get_piece(cur_x, cur_y)
        return False

    def _get_moves_at_diagonal(
            self,
            x: int,
            y: int,
            delta_x: int,
            delta_y: int,
            jumped_pieces: set[Piece] = None,
            is_only_jump: bool = False,
            maybe_piece: Piece | None = None) -> list[list[PieceMoveWithCapture]]:
        """Returns the possible moves at the diagonal. If `isOnlyJump` (default false), then only single capture moves
        will be returned."""
        piece = maybe_piece or self._pieces[x][y]
        if not piece:
            return []
        # If piece is a normal piece, and not can take backwards, then return. Piece can't go backwards.
        backwards = is_backwards(piece, delta_x, delta_y)
        if piece.type == 'm' and not self.config['canTakeBackwards'] and backwards:
            return []

        to_piece = self._get_piece(x + delta_x, y + delta_y)
        if to_piece is None:
            return []  # to_piece is not on the board. This is an invalid move.
        elif not to_piece or to_piece in jumped_pieces:
            # No piece at `to`. This is a regular "move". Non-kings can only move if they haven't already captured.
            has_captured = len(jumped_pieces) > 1
            if backwards or (piece.type == 'm' and has_captured):
                return []

            can_capture = self._can_capture(piece, x, y, delta_x, delta_y)
            if (is_only_jump or (self._has_capture and not has_captured)) and not can_capture:
                # This piece needs to capture something, but can't. Return.
                return []

            if has_captured and (piece.type != 'k' or not self.config['hasFlyingKings']):
                # This piece has already captured, and it's not a flying king. Return.
                return []

            next_move: PieceMoveWithCapture = PieceMoveWithCapture(
                frm=(x, y),
                to=(x + delta_x, y + delta_y),
                captures=len(jumped_pieces) - 1,
                is_capture=False,
                team_index=piece.team_index,
            )
            # If this piece is a king and can "fly", then keep going.
            if piece.type == 'k' and self.config['hasFlyingKings']:
                return self._get_continued_moves(
                    next_move,
                    jumped_pieces,
                    piece,
                    is_only_jump,
                    [MovementDelta(delta_x, delta_y)])
            # No piece at the to location. We can move here!
            return [[next_move]]
        elif to_piece.team_index == piece.team_index:
            # Piece at `to` is on the same team. This is an invalid move.
            return []

        # At this point, the diagonal holds an enemy piece. Check to see if the next diagonal is free. If so, we can
        # move here!
        next_piece = self._get_piece(x + delta_x * 2, y + delta_y * 2)
        if next_piece is not 0:
            return []  # Can't move here. Piece is in the way.
        elif backwards and not self.config['canTakeBackwards']:
            return []  # Piece can't take backwards.

        # This location has a gap, and we can capture here. Keep going to see if we can make more moves.
        next_move: PieceMoveWithCapture = PieceMoveWithCapture(
            frm=(x, y),
            to=(x + delta_x * 2, y + delta_y * 2),
            captures=len(jumped_pieces),  # Full length of jumpedPieces because we're adding toPiece down below.
            is_capture=True,
            team_index=piece.team_index)
        if (not self.config['canChainTake']) or is_only_jump:
            return [[next_move]]

        new_jumped_pieces = set(jumped_pieces)
        new_jumped_pieces.add(to_piece)
        return self._get_continued_moves_for_direction(
            next_move,
            new_jumped_pieces,
            piece,
            delta_x,
            delta_y,
            is_only_jump)

    def _get_continued_moves_for_direction(
            self,
            next_move: PieceMoveWithCapture,
            jumped_pieces: set[Piece],
            piece: Piece,
            delta_x: int,
            delta_y: int,
            is_only_jump: bool) -> list[list[PieceMoveWithCapture]]:
        """Returns all the possible moves that can be made after the `next_move`."""
        # Prevent backwards jumps
        deltas = [delta for delta in MOVEMENT_DELTAS if delta.delta_x != -1 * delta_x or delta.delta_y != -1 * delta_y]
        result = self._get_continued_moves(next_move, jumped_pieces, piece, is_only_jump, deltas)

        for idx, moves in enumerate(result):
            if len(moves) < 2:
                continue  # Moves is short, and is guaranteed to be a capture.
            can_turn = True
            last_delta_x = delta_x
            last_delta_y = delta_y
            last_valid_move_index = 0
            for i in range(1, len(moves)):
                if moves[i].captures != moves[i - 1].captures:
                    can_turn = True
                    last_valid_move_index = i
                    continue  # This was a capture!
                # If piece is turning, then they must make a capture before turning again. Additionally, they must make
                # a capture during this turn for it to count.
                if not is_same_direction(moves[i].frm, moves[i].to, last_delta_x, last_delta_y):
                    if not can_turn:
                        break
                    last_valid_move_index = i - 1
                    can_turn = False
                    last_delta_x = (moves[i].to[0] - moves[i].frm[0]) / 2
                    last_delta_y = (moves[i].to[1] - moves[i].frm[1]) / 2
                elif can_turn:
                    last_valid_move_index = i
            result[idx] = moves[:last_valid_move_index + 1]
        return result

    def _get_continued_moves(
            self,
            next_move: PieceMoveWithCapture,
            jumped_pieces: set[Piece],
            piece: Piece,
            is_only_jump: bool,
            directions=MOVEMENT_DELTAS):
        # Deep copy of the piece and modifying its type if it's promotion
        piece_copy = copy(piece)
        piece_copy_promotion = copy(piece)
        piece_copy_promotion.to = piece_copy_promotion.to
        piece_move = PieceMovePartial(team_index=piece_copy.team_index, to=(piece_copy.x, piece_copy.y))
        piece_copy.type = 'k' if self._is_promotion(piece_move, False) else piece.type

        chains = self._get_chained_moves_impl(
            next_move.to[0],
            next_move.to[1],
            jumped_pieces,
            piece_copy,
            is_only_jump,
            directions)

        result = [] if self.config['mustForceTake'] else [[next_move]]
        for chain in chains:
            result.append([next_move, *chain])

        # If the player is required to take, then they must take the longest chain.
        # noinspection PyChainedComparisons
        if self.config['mustForceTake'] and len(result) <= 0 and len(jumped_pieces) > 0:
            result.append([next_move])

        # Return all the possible moves.
        return result

    def _is_promotion(self, piece_move: PieceMovePartial, is_last: bool):
        """
        Returns `true` if `piece_move` has reached the opposing team's side and should
        be promoted. `is_last` is true if `piece_move` is landing here as its final
        move in a chain of jumps. In some game modes, such as International
        Draughts, a piece is only promoted if `is_last` is true.
        """
        if not self.config['isPromotedImmediately'] and not is_last:
            return False  # Can't promote piece while it's still jumping.

        team_index = piece_move.team_index
        columns, rows = self.config['field'], self.config['field']

        if team_index == 0:
            return piece_move.to[1] == rows - 1
        elif team_index == 1:
            return piece_move.to[1] == 0
        elif team_index == 2:
            return piece_move.to[0] == columns - 1
        return piece_move.to[0] == 0

    def _index_to_coordinate(self, index: int) -> tuple[int, int]:
        """Returns the 0-based (x, y) coordinate from the PDN index."""
        columns = self._get_dimensions().columns
        is_odd_y = ((index - 1) % columns) >= (columns / 2)
        piece_index = index * 2 - 1 - (1 if is_odd_y else 0)
        return piece_index % columns, piece_index // columns

    def _get_dimensions(self) -> Dimensions:
        return Dimensions(rows=self.config['field'], columns=self.config['field'])

    def get_coordinate_from_square(self, square: str) -> tuple[int, int]:
        """Returns the (x, y) coordinate of the square on the board."""
        column, row = square[0], square[1:]
        x = BOARD_COLUMNS.index(column)
        y = self.config['field'] - int(row)
        return x, y

    def _has_threefold_repetition(self) -> bool:
        """Check if there is a threefold repetition."""
        if not self.config['threefoldRepetition']:
            return False
        state = self._get_state_string()
        result = 1 + self._game_state_counts.get(state, 0)
        self._game_state_counts[state] = result
        return result >= 3

    def _get_state_string(self) -> str:
        """Returns the board state as a string."""
        piece_state = '|'.join(f"{piece.type},{piece.square}" for piece in self._pieces if piece)
        return f"{self._current_team_index}{piece_state}"

    def _get_piece(self, x: int, y: int) -> Piece | None | Literal[0]:
        """Returns the piece at (x, y) (0-based), or null if there's no piece at that location. Returns `None` if (x, y)
        is off the board."""
        if x < 0 or y < 0:
            return None
        return self._pieces[x][y] if x >= len(self._pieces) or y >= len(self._pieces[x]) else 0

    def _set_piece(self, x: int, y: int, piece: TypeTeamIndex | dict | None | Literal[0]):
        self._pieces[x][y] =\
            Piece(**piece, x=x, y=y, square=self._get_square_from_coordinate(x, y)) if piece else 0

    def _get_square_from_coordinate(self, x: int, y: int) -> str:
        dimensions = self._get_dimensions()
        columns, rows = dimensions.columns, dimensions.rows
        if x < 0 or x >= columns or y < 0 or y >= rows:
            raise ValueError(
                f"Expected getSquareFromCoordinate to be in 0-{columns - 1} range for x and y. Got: ({x}, {y})")
        return BOARD_COLUMNS[x] + (rows - y)

    def _get_team_piece_indices(self) -> np.array:
        """Returns all the pieces for the current team."""
        x_indices, y_indices = np.where(self._pieces == self._current_team_index)
        return zip(x_indices, y_indices)


def is_same_direction(frm: tuple[int, int], to: tuple[int, int], delta_x: int, delta_y: int) -> bool:
    """Returns `true` if moving `from` to `to` goes in the same direction as `delta_x`, `delta_y`."""
    from_x, from_y = frm
    to_x, to_y = to
    if delta_x < 0 and from_x <= to_x:
        return False
    elif delta_x > 0 and from_x >= to_x:
        return False
    elif delta_y < 0 and from_y <= to_y:
        return False
    return from_y < to_y


def is_backwards(piece: Piece, delta_x: int, delta_y: int):
    """
    Returns true if (deltaX, deltaY) would result in "backwards" movement for
    `piece`. Backwards movement isn't normally allowed unless back-taking is
    allowed. Kings don't have backwards movement (they can move anywhere).
    """
    team_index, type_ = piece.team_index, piece.type
    if type_ == 'k':
        return False

    if team_index == 0:
        return delta_y < 0  # Backwards is up
    elif team_index == 1:
        return delta_y > 0  # Backwards is down
    elif team_index == 2:
        return delta_x < 0  # Backwards is to the left
    return delta_x > 0  # teamIndex === 3 (backwards is to the right)
