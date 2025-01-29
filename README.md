# Chess Project

Welcome to my Chess engine! This engine can challenge players with an ELO rating of 1000 or higher, having beaten the Maria 1000 ELO chess bot on chess.com.

## How It Works

The Chess Project utilizes advanced algorithms and heuristics to evaluate board positions and make strategic decisions. The engine is capable of:

- **Evaluating potential moves using a minimax algorithm with alpha-beta pruning**:
  - The minimax algorithm is a recursive method used in decision-making and game theory. It evaluates the possible moves in a game by simulating all potential future moves and counter-moves, aiming to minimize the possible loss for a worst-case scenario.
  - Alpha-beta pruning is an optimization technique for the minimax algorithm. It reduces the number of nodes evaluated in the search tree by eliminating branches that cannot possibly influence the final decision. This makes the algorithm more efficient and faster.

- **Implementing various opening strategies to gain an early advantage**:
  - Opening strategies are predefined sequences of moves used at the beginning of a chess game. These strategies are designed to control the center of the board, develop pieces to optimal positions, and ensure the safety of the king.
  - The engine has a database of popular and effective opening moves, allowing it to play strong openings and respond appropriately to the opponent's moves.

- **Adapting to different play styles and tactics**:
  - The engine can adjust its strategy based on the opponent's play style, whether aggressive, defensive, or balanced. It uses heuristics to evaluate the board and determine the best course of action.
  - Tactics such as forks, pins, and skewers are recognized and utilized by the engine to gain material advantage or improve its position.

- **Using FEN notation for easy input**:
    - The engine supports Forsyth-Edwards Notation (FEN), a standard notation for describing a particular board position of a chess game. This allows users to input and load specific board states easily.
    - FEN strings provide a concise way to represent the placement of all pieces, the active player, castling availability, en passant target squares, halfmove clock, and fullmove number.
    - By using FEN notation, users can set up custom positions, analyze specific scenarios, or continue games from a particular point, enhancing the flexibility and usability of the engine.

Overall, the combination of these advanced algorithms and strategic implementations allows my chess engine to play at a high (ish) level, making it a challenging opponent for human players and other chess engines.
