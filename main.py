from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set, NamedTuple
import copy
from functools import lru_cache
from collections import defaultdict

BOARD_SIZE = 8

def create_standard_board():
    initial_pieces = []
    
    # Set up pawns
    for x in range(BOARD_SIZE):
        initial_pieces.append(Piece(Team.WHITE, PieceType.PAWN, Position(x, 1)))
        initial_pieces.append(Piece(Team.BLACK, PieceType.PAWN, Position(x, 6)))
    
    # Set up other pieces
    piece_order = [
        PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
        PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
    ]
    
    for x in range(BOARD_SIZE):
        initial_pieces.append(Piece(Team.WHITE, piece_order[x], Position(x, 0)))
        initial_pieces.append(Piece(Team.BLACK, piece_order[x], Position(x, 7)))
    
    return Board(initial_pieces)

def create_board_from_fen(fen: str) -> 'Board':
    """Create a chess board from FEN notation."""
    
    # Get active color from FEN (after the board position part)
    parts = fen.split(' ')

    pieces = []
    ranks = parts[0].split('/')[0:8]  # Get board position part
    
    piece_map = {
        'p': (Team.BLACK, PieceType.PAWN),
        'n': (Team.BLACK, PieceType.KNIGHT),
        'b': (Team.BLACK, PieceType.BISHOP),
        'r': (Team.BLACK, PieceType.ROOK),
        'q': (Team.BLACK, PieceType.QUEEN),
        'k': (Team.BLACK, PieceType.KING),
        'P': (Team.WHITE, PieceType.PAWN),
        'N': (Team.WHITE, PieceType.KNIGHT),
        'B': (Team.WHITE, PieceType.BISHOP),
        'R': (Team.WHITE, PieceType.ROOK),
        'Q': (Team.WHITE, PieceType.QUEEN),
        'K': (Team.WHITE, PieceType.KING)
    }
    
    for y, rank in enumerate(reversed(ranks)):
        x = 0
        for char in rank:
            if char.isdigit():
                x += int(char)
            else:
                team, piece_type = piece_map[char]
                pieces.append(Piece(team, piece_type, Position(x, y)))
                x += 1

    if len(parts) >= 2:
        active_color = parts[1]
        if active_color == 'b':
            current_turn = Team.BLACK
        else:  # 'w'
            current_turn = Team.WHITE
    else:
        current_turn = Team.WHITE  # Default to white if not specified

    return Board(pieces), current_turn

class GamePhase(IntEnum):
    OPENING = 0
    MIDDLEGAME = 1
    ENDGAME = 2

class Team(IntEnum):
    BLACK = 0
    WHITE = 1

class PieceType(IntEnum):
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5

    @staticmethod
    @lru_cache(maxsize=None)
    def get_piece_name(piece_type: int) -> str:
        return {
            0: "Pawn", 1: "Knight", 2: "Bishop",
            3: "Rook", 4: "Queen", 5: "King"
        }[piece_type]

class Move(NamedTuple):
    piece: 'Piece'
    target: 'Position'
    captured: Optional['Piece'] = None
    score: float = 0.0

@dataclass(frozen=True)
class Position:
    x: int
    y: int
    
    def move(self, dx: int, dy: int) -> 'Position':
        return Position(self.x + dx, self.y + dy)
    
    def is_valid(self) -> bool:
        return 0 <= self.x < 8 and 0 <= self.y < 8

    @staticmethod
    @lru_cache(maxsize=64)
    def to_algebraic(pos: 'Position') -> str:
        return f"{'abcdefgh'[pos.x]}{pos.y + 1}"

class Piece:
    # Precomputed move patterns
    _KNIGHT_MOVES = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    _KING_MOVES = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    _DIAGONAL_MOVES = tuple((i, i) for i in range(-7, 8) if i != 0) + tuple((i, -i) for i in range(-7, 8) if i != 0)
    _STRAIGHT_MOVES = tuple((0, i) for i in range(-7, 8) if i != 0) + tuple((i, 0) for i in range(-7, 8) if i != 0)

    _MOVE_CACHE = {}

    # Piece-square tables for positional evaluation
    _PST = {
        PieceType.PAWN: [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ],
        PieceType.KNIGHT: [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ]
    }

    def __init__(self, team: Team, type: PieceType, pos: Position):
        self.team = team
        self.type = type
        self.pos = pos
        self.has_moved = False
        self._move_patterns = self._init_move_patterns()
        
        # Check if piece is on its starting square
        start_y = 1 if team == Team.WHITE and type == PieceType.PAWN else \
              6 if team == Team.BLACK and type == PieceType.PAWN else \
              0 if team == Team.WHITE else 7
        self.has_moved = not (pos.y == start_y and (
            (type == PieceType.PAWN) or
            (type == PieceType.ROOK and pos.x in (0, 7)) or
            (type == PieceType.KNIGHT and pos.x in (1, 6)) or
            (type == PieceType.BISHOP and pos.x in (2, 5)) or
            (type == PieceType.QUEEN and pos.x == 3) or
            (type == PieceType.KING and pos.x == 4)
        ))

    def _init_move_patterns(self) -> Tuple[Tuple[int, int], ...]:
        if self.type == PieceType.PAWN:
            direction = 1 if self.team == Team.WHITE else -1
            return ((0, direction), (0, 2 * direction), (-1, direction), (1, direction))
        return {
            PieceType.KNIGHT: self._KNIGHT_MOVES,
            PieceType.BISHOP: self._DIAGONAL_MOVES,
            PieceType.ROOK: self._STRAIGHT_MOVES,
            PieceType.QUEEN: self._DIAGONAL_MOVES + self._STRAIGHT_MOVES,
            PieceType.KING: self._KING_MOVES
        }[self.type]

    def get_valid_moves(self, board: 'Board', check_for_check: bool = True) -> List[Move]:
            valid_moves = []
            
            for dx, dy in self._move_patterns:
                new_pos = self.pos.move(dx, dy)
                if not new_pos.is_valid():
                    continue
                    
                if self.type == PieceType.PAWN:
                    if not self._validate_pawn_move(new_pos, dx, dy, board):
                        continue
                elif not self._validate_move(new_pos, board):
                    continue
                
                captured = board.get_piece_at(new_pos)
                move = Move(self, new_pos, captured)
                
                if check_for_check:
                    # Skip moves that leave king in check
                    temp_board = board.make_move(move)
                    if not temp_board.is_in_check(self.team):
                        valid_moves.append(move)
                else:
                    valid_moves.append(move)
            
            return valid_moves

    def _validate_pawn_move(self, new_pos: Position, dx: int, dy: int, board: 'Board') -> bool:
        if dx == 0:  # Forward move
            if board.get_piece_at(new_pos) is not None:
                return False
            if abs(dy) == 2:
                if self.has_moved:
                    return False
                mid_pos = self.pos.move(0, dy // 2)
                if board.get_piece_at(mid_pos) is not None:
                    return False
        else:  # Diagonal capture
            target = board.get_piece_at(new_pos)
            if target is None or target.team == self.team:
                return False
        return True

    def _validate_move(self, new_pos: Position, board: 'Board') -> bool:
        target = board.get_piece_at(new_pos)
        if target is not None and target.team == self.team:
            return False
            
        if self.type not in (PieceType.KNIGHT, PieceType.KING):
            return self._is_path_clear(new_pos, board)
        return True

    def _is_path_clear(self, target: Position, board: 'Board') -> bool:
        dx = target.x - self.pos.x
        dy = target.y - self.pos.y
        
        if dx == 0:
            step = dy // abs(dy)
            return all(board.get_piece_at(Position(self.pos.x, y)) is None 
                      for y in range(self.pos.y + step, target.y, step))
        
        if dy == 0:
            step = dx // abs(dx)
            return all(board.get_piece_at(Position(x, self.pos.y)) is None 
                      for x in range(self.pos.x + step, target.x, step))
        
        if abs(dx) == abs(dy):
            step_x = dx // abs(dx)
            step_y = dy // abs(dy)
            return all(board.get_piece_at(Position(self.pos.x + i * step_x, 
                                                 self.pos.y + i * step_y)) is None 
                      for i in range(1, abs(dx)))
        return False

class Board:
    def __init__(self, pieces: List[Piece]):
        self.pieces = pieces
        self._position_cache = {piece.pos: piece for piece in pieces}
        self._material_count = self._count_material()
        
    def _count_material(self) -> Dict[Team, int]:
        counts = defaultdict(int)
        for piece in self.pieces:
            if piece.type != PieceType.KING:
                counts[piece.team] += ChessEngine.PIECE_VALUES[piece.type]
        return counts

    def get_game_phase(self) -> GamePhase:
        total_material = sum(self._material_count.values())
        if total_material > 6000:  # Both sides have most pieces
            return GamePhase.OPENING
        elif total_material < 3000:  # Few pieces remain
            return GamePhase.ENDGAME
        return GamePhase.MIDDLEGAME

    def get_piece_at(self, pos: Position) -> Optional[Piece]:
        return self._position_cache.get(pos)

    def make_move(self, move: Move) -> 'Board':
        new_pieces = [p for p in self.pieces if p != move.piece and p != move.captured]
        moved_piece = Piece(move.piece.team, move.piece.type, move.target)
        moved_piece.has_moved = True
        new_pieces.append(moved_piece)
        return Board(new_pieces)

    def get_all_moves(self, team: Team) -> List[Move]:
        return [move for piece in self.pieces if piece.team == team 
                for move in piece.get_valid_moves(self)]
    def is_in_check(self, team: Team) -> bool:
        king = next((p for p in self.pieces if p.team == team and p.type == PieceType.KING), None)
        if not king:
            return False
            
        return any(king.pos == move.target 
                  for p in self.pieces if p.team != team
                  for move in p.get_valid_moves(self, check_for_check=False))

    def is_checkmate(self, team: Team) -> bool:
        return self.is_in_check(team) and not self.get_all_moves(team)

class ChessEngine:
    # Material values
    PIECE_VALUES = {
        PieceType.PAWN: 100,
        PieceType.KNIGHT: 320,
        PieceType.BISHOP: 330,
        PieceType.ROOK: 500,
        PieceType.QUEEN: 900,
        PieceType.KING: 20000
    }

    def __init__(self, board: Board, max_depth: int = 4):
        self.board = board
        self.max_depth = max_depth
        self._transposition_table: Dict[str, Tuple[float, int]] = {}
        self._killer_moves: List[List[Optional[Move]]] = [[None, None] for _ in range(max_depth)]
        
    def find_best_move(self, team: Team) -> Optional[Move]:
        if self.board.is_checkmate(team):
            return None
            
        moves = self._get_sorted_moves(team)
        if not moves:
            return None
            
        best_move = moves[0]
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in moves:
            # Make move
            new_board = self.board.make_move(move)
            
            # Search position
            value = -self._negamax(new_board, self.max_depth - 1, -beta, -alpha, Team(1 - team))
            
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            
            print(f"{Position.to_algebraic(move.target)} - {PieceType.get_piece_name(move.piece.type)}: {value}")
            
        return best_move

    def _get_sorted_moves(self, team: Team) -> List[Move]:
        """Get and sort moves using MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)"""
        moves = self.board.get_all_moves(team)
        
        # Score moves
        for move in moves:
            score = 0
            # Capture value
            if move.captured:
                score = 10 * self.PIECE_VALUES[move.captured.type] - self.PIECE_VALUES[move.piece.type]
            # Position value from piece-square tables
            if move.piece.type in Piece._PST:
                y = move.target.y if team == Team.WHITE else 7 - move.target.y
                score += Piece._PST[move.piece.type][y][move.target.x] * 0.1
            move = move._replace(score=score)
            
        return sorted(moves, key=lambda m: m.score, reverse=True)

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float, team: Team) -> float:
        # Check transposition table
        board_hash = str([(p.team, p.type, p.pos.x, p.pos.y) for p in board.pieces])
        if board_hash in self._transposition_table:
            stored_value, stored_depth = self._transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_value
        
        if depth == 0:
            return self._evaluate_position(board, team)
            
        value = float('-inf')
        moves = self._get_sorted_moves(team)
        
        if not moves:
            if board.is_in_check(team):
                return -900000  # Checkmate
            return 0  # Stalemate
            
        for move in moves:
            new_board = board.make_move(move)
            value = max(value, -self._negamax(new_board, depth - 1, -beta, -alpha, Team(1 - team)))
            alpha = max(alpha, value)
            
            if alpha >= beta:
                # Store killer move
                if not move.captured:
                    self._killer_moves[depth][1] = self._killer_moves[depth][0]
                    self._killer_moves[depth][0] = move
                break
        
        # Store position in transposition table
        self._transposition_table[board_hash] = (value, depth)
        return value

    def _evaluate_position(self, board: Board, team: Team) -> float:
        if board.is_checkmate(team):
            return -900000
        elif board.is_checkmate(Team(1 - team)):
            return 900000

        score = 0.0
        game_phase = board.get_game_phase()
        
        for piece in board.pieces:
            multiplier = 1 if piece.team == team else -1
            
            # Material value
            score += self.PIECE_VALUES[piece.type] * multiplier
            
            # Piece-specific positional bonuses
            if piece.type == PieceType.PAWN:
                score += self._evaluate_pawn(piece, game_phase) * multiplier
            elif piece.type == PieceType.KNIGHT:
                score += self._evaluate_knight(piece, game_phase) * multiplier
            elif piece.type == PieceType.BISHOP:
                score += self._evaluate_bishop(piece, game_phase) * multiplier
            elif piece.type == PieceType.ROOK:
                score += self._evaluate_rook(piece, game_phase) * multiplier
            elif piece.type == PieceType.QUEEN:
                score += self._evaluate_queen(piece, game_phase) * multiplier
            elif piece.type == PieceType.KING:
                score += self._evaluate_king(piece, game_phase) * multiplier
        
        # Evaluate piece mobility
        mobility_score = self._evaluate_mobility(board, team)
        score += mobility_score * (0.1 if game_phase != GamePhase.ENDGAME else 0.2)
        
        # Evaluate pawn structure
        pawn_structure_score = self._evaluate_pawn_structure(board, team)
        score += pawn_structure_score
        
        # Evaluate king safety
        if game_phase != GamePhase.ENDGAME:
            king_safety_score = self._evaluate_king_safety(board, team)
            score += king_safety_score * (1.5 if game_phase == GamePhase.OPENING else 1.0)
        
        return score

    def _evaluate_pawn(self, pawn: Piece, phase: GamePhase) -> float:
        score = 0.0
        
        # Pawn advancement
        rank = pawn.pos.y if pawn.team == Team.WHITE else 7 - pawn.pos.y
        score += 10 * rank
        
        # Central pawns
        if 2 <= pawn.pos.x <= 5:
            score += 20 if (3 <= pawn.pos.x <= 4) else 10
        
        # Passed pawn bonus (increasingly valuable in endgame)
        if self._is_passed_pawn(pawn):
            score += 50 * (2 if phase == GamePhase.ENDGAME else 1)
        
        return score

    def _evaluate_knight(self, knight: Piece, phase: GamePhase) -> float:
        score = 0.0
        
        # Central control
        if 2 <= knight.pos.x <= 5 and 2 <= knight.pos.y <= 5:
            score += 30 if (3 <= knight.pos.x <= 4 and 3 <= knight.pos.y <= 4) else 20
        
        # Outpost bonus (knight protected by pawn and can't be attacked by enemy pawns)
        if self._is_outpost(knight):
            score += 40
        
        return score

    def _evaluate_bishop(self, bishop: Piece, phase: GamePhase) -> float:
        score = 0.0
        
        # Diagonal control
        controlled_squares = self._get_controlled_squares(bishop)
        score += len(controlled_squares) * 5
        
        # Bishop pair bonus
        if self._has_bishop_pair(bishop.team):
            score += 50
        
        # Penalty for blocked bishops in opening
        if phase == GamePhase.OPENING and self._is_blocked_bishop(bishop):
            score -= 30
        
        return score

    def _evaluate_rook(self, rook: Piece, phase: GamePhase) -> float:
        score = 0.0
        
        # Open file bonus
        if self._is_open_file(rook):
            score += 40
        elif self._is_semi_open_file(rook):
            score += 20
        
        # Rook on seventh rank
        seventh_rank = 6 if rook.team == Team.WHITE else 1
        if rook.pos.y == seventh_rank:
            score += 30
        
        # Connected rooks
        if self._are_rooks_connected(rook):
            score += 20
        
        return score

    def _evaluate_queen(self, queen: Piece, phase: GamePhase) -> float:
        score = 0.0
        
        # Early development penalty
        if phase == GamePhase.OPENING:
            if (queen.team == Team.WHITE and queen.pos.y < 2) or \
               (queen.team == Team.BLACK and queen.pos.y > 5):
                score -= 30
        
        # Central control
        controlled_squares = self._get_controlled_squares(queen)
        score += len(controlled_squares) * 2
        
        return score

    def _evaluate_king(self, king: Piece, phase: GamePhase) -> float:
        score = 0.0
        
        if phase == GamePhase.ENDGAME:
            # King centralization in endgame
            center_dist = abs(3.5 - king.pos.x) + abs(3.5 - king.pos.y)
            score -= 10 * center_dist
        else:
            # King safety in opening/middlegame
            if self._is_castled(king):
                score += 60
        
        return score

    def _is_castled(self, king: Piece) -> bool:
        """Check if king has castled based on its position"""
        if king.team == Team.WHITE:
            return king.has_moved and (king.pos.x <= 2 or king.pos.x >= 6)
        else:
            return king.has_moved and (king.pos.x <= 2 or king.pos.x >= 6)
            
            # Pawn shield
            pawn_shield_score = self._evaluate_pawn_shield(king)
            score += pawn_shield_score
        
        return score

    def _evaluate_mobility(self, board: Board, team: Team) -> float:
        """Evaluate piece mobility (number of legal moves)"""
        mobility = len(board.get_all_moves(team)) - len(board.get_all_moves(Team(1 - team)))
        return mobility * 5

    def _evaluate_pawn_structure(self, board: Board, team: Team) -> float:
        """Evaluate pawn structure (doubled, isolated, backward pawns)"""
        score = 0.0
        pawns = [p for p in board.pieces if p.team == team and p.type == PieceType.PAWN]
        
        # Count pawns on each file
        files = defaultdict(int)
        for pawn in pawns:
            files[pawn.pos.x] += 1
        
        for pawn in pawns:
            # Doubled pawns penalty
            if files[pawn.pos.x] > 1:
                score -= 20
            
            # Isolated pawns penalty
            if not any(files[x] for x in (pawn.pos.x - 1, pawn.pos.x + 1) if 0 <= x < 8):
                score -= 30
            
            # Backward pawns penalty
            if self._is_backward_pawn(pawn, pawns):
                score -= 25
        
        return score

    def _is_backward_pawn(self, pawn: Piece, friendly_pawns: List[Piece]) -> bool:
        """Check if pawn is backward (can't be protected by adjacent pawns and blocked by enemy pawn)"""
        direction = 1 if pawn.team == Team.WHITE else -1
        
        # Check if pawn can be protected by friendly pawns
        can_be_protected = any(p for p in friendly_pawns
                             if abs(p.pos.x - pawn.pos.x) == 1
                             and p.pos.y == pawn.pos.y)
        
        if can_be_protected:
            return False
            
        # Check if blocked by enemy pawn
        enemy_pawns = [p for p in self.board.pieces
                      if p.type == PieceType.PAWN and p.team != pawn.team]
        
        return any(p for p in enemy_pawns
                  if p.pos.x == pawn.pos.x
                  and ((pawn.team == Team.WHITE and p.pos.y > pawn.pos.y)
                       or (pawn.team == Team.BLACK and p.pos.y < pawn.pos.y)))

    def _evaluate_king_safety(self, board: Board, team: Team) -> float:
        """Evaluate king safety based on pawn shield and attacking pieces"""
        score = 0.0
        king = next(p for p in board.pieces if p.team == team and p.type == PieceType.KING)
        
        # Count attacking pieces near king
        king_zone = {Position(king.pos.x + dx, king.pos.y + dy)
                    for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                    if 0 <= king.pos.x + dx < 8 and 0 <= king.pos.y + dy < 8}
        
        attackers = sum(1 for p in board.pieces 
                       if p.team != team and any(move.target in king_zone 
                                               for move in p.get_valid_moves(board, check_for_check=False)))
        score -= attackers * 20
        
        # Evaluate pawn shield
        shield_bonus = self._evaluate_pawn_shield(king)
        score += shield_bonus
        
        return score

    def _evaluate_pawn_shield(self, king: Piece) -> float:
        """Evaluate the pawn protection in front of the king"""
        score = 0.0
        
        # Define pawn shield positions relative to king
        if king.team == Team.WHITE:
            shield_positions = [
                Position(king.pos.x + dx, king.pos.y + 1)
                for dx in [-1, 0, 1]
                if 0 <= king.pos.x + dx < 8 and king.pos.y + 1 < 8
            ]
        else:
            shield_positions = [
                Position(king.pos.x + dx, king.pos.y - 1)
                for dx in [-1, 0, 1]
                if 0 <= king.pos.x + dx < 8 and king.pos.y - 1 >= 0
            ]
            
        # Score pawns in shield positions
        for pos in shield_positions:
            piece = self.board.get_piece_at(pos)
            if piece and piece.type == PieceType.PAWN and piece.team == king.team:
                score += 10
                
        return score

    def _is_passed_pawn(self, pawn: Piece) -> bool:
        """Check if pawn is passed (no enemy pawns ahead on same or adjacent files)"""
        direction = 1 if pawn.team == Team.WHITE else -1
        target_ranks = range(pawn.pos.y + direction, 8 if direction > 0 else -1, direction)
        
        for x in (pawn.pos.x - 1, pawn.pos.x, pawn.pos.x + 1):
            if not 0 <= x < 8:
                continue
            for y in target_ranks:
                piece = self.board.get_piece_at(Position(x, y))
                if piece and piece.type == PieceType.PAWN and piece.team != pawn.team:
                    return False
        return True

    def _is_outpost(self, knight: Piece) -> bool:
        """Check if knight is on an outpost (protected by friendly pawn, can't be attacked by enemy pawns)"""
        if knight.team == Team.WHITE:
            if knight.pos.y < 4:  # Must be on opponent's half
                return False
            # Check if protected by friendly pawn
            if not any(p for p in self.board.pieces 
                      if p.team == Team.WHITE and p.type == PieceType.PAWN
                      and abs(p.pos.x - knight.pos.x) == 1 and p.pos.y == knight.pos.y - 1):
                return False
            # Check if can be attacked by enemy pawns
            if any(p for p in self.board.pieces
                  if p.team == Team.BLACK and p.type == PieceType.PAWN
                  and abs(p.pos.x - knight.pos.x) == 1 and p.pos.y > knight.pos.y):
                return False
            return True
        else:
            # Similar logic for black knight
            if knight.pos.y > 3:
                return False
            if not any(p for p in self.board.pieces 
                      if p.team == Team.BLACK and p.type == PieceType.PAWN
                      and abs(p.pos.x - knight.pos.x) == 1 and p.pos.y == knight.pos.y + 1):
                return False
            if any(p for p in self.board.pieces
                  if p.team == Team.WHITE and p.type == PieceType.PAWN
                  and abs(p.pos.x - knight.pos.x) == 1 and p.pos.y < knight.pos.y):
                return False
            return True

    # Add implementation for helper methods used in evaluation
    def _get_controlled_squares(self, piece: Piece) -> Set[Position]:
        """Get all squares that a piece controls/attacks"""
        controlled = set()
        moves = piece.get_valid_moves(self.board)
        for move in moves:
            controlled.add(move.target)
        return controlled

    def _has_bishop_pair(self, team: Team) -> bool:
        """Check if a team has both bishops"""
        bishops = [p for p in self.board.pieces if p.team == team and p.type == PieceType.BISHOP]
        if len(bishops) != 2:
            return False
        # Check if bishops are on opposite colors
        return (bishops[0].pos.x + bishops[0].pos.y) % 2 != (bishops[1].pos.x + bishops[1].pos.y) % 2

    def _is_blocked_bishop(self, bishop: Piece) -> bool:
        """Check if bishop is blocked by own pawns"""
        if bishop.team == Team.WHITE:
            return any(p for p in self.board.pieces
                      if p.team == Team.WHITE and p.type == PieceType.PAWN
                      and p.pos.y == bishop.pos.y + 1
                      and abs(p.pos.x - bishop.pos.x) <= 1)
        else:
            return any(p for p in self.board.pieces
                      if p.team == Team.BLACK and p.type == PieceType.PAWN
                      and p.pos.y == bishop.pos.y - 1
                      and abs(p.pos.x - bishop.pos.x) <= 1)

    def _is_open_file(self, rook: Piece) -> bool:
        """Check if rook is on an open file (no pawns)"""
        return not any(p for p in self.board.pieces
                      if p.type == PieceType.PAWN and p.pos.x == rook.pos.x)

    def _is_semi_open_file(self, rook: Piece) -> bool:
        """Check if rook is on a semi-open file (no friendly pawns)"""
        return not any(p for p in self.board.pieces
                      if p.type == PieceType.PAWN and p.team == rook.team
                      and p.pos.x == rook.pos.x)

    def _are_rooks_connected(self, rook: Piece) -> bool:
        """Check if rooks are connected (on same rank/file with no pieces between)"""
        other_rook = next((p for p in self.board.pieces
                          if p.team == rook.team and p.type == PieceType.ROOK
                          and p != rook), None)
        if not other_rook:
            return False
            
        if rook.pos.x == other_rook.pos.x:
            min_y = min(rook.pos.y, other_rook.pos.y)
            max_y = max(rook.pos.y, other_rook.pos.y)
            return not any(p for p in self.board.pieces
                         if p.pos.x == rook.pos.x
                         and min_y < p.pos.y < max_y)
        elif rook.pos.y == other_rook.pos.y:
            min_x = min(rook.pos.x, other_rook.pos.x)
            max_x = max(rook.pos.x, other_rook.pos.x)
            return not any(p for p in self.board.pieces
                         if p.pos.y == rook.pos.y
                         and min_x < p.pos.x < max_x)
        return False

def main():
    board, team = create_board_from_fen(input())
    engine = ChessEngine(board, max_depth=2)
    best_move = engine.find_best_move(team)
    
    if best_move:
        print(f"Best move: {PieceType.get_piece_name(best_move.piece.type)} to",
              Position.to_algebraic(best_move.target))
    else:
        print("No valid moves available")

if __name__ == "__main__":
    main()