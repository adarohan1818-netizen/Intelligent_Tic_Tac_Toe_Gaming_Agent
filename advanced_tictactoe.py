import random
import math
import copy
from typing import List, Tuple, Dict, Any, Optional

# =====================================================================
# CO1: PROBLEM REPRESENTATION & STATE SPACE FORMULATION
# Formulating Tic-Tac-Toe as an AI problem.
# State Space: Grid configuration, current player, and action history.
# =====================================================================
class TicTacToeState:
    def __init__(self, size: int = 3, board: Optional[List[List[str]]] = None, current_player: str = 'X'):
        self.size = size
        # State representation using core Python data structures (lists)
        self.board = board if board else [[' ' for _ in range(size)] for _ in range(size)]
        self.current_player = current_player  # 'X' or 'O'
        self.history = []  # For traceable reasoning

    def get_actions(self) -> List[Tuple[int, int]]:
        """Returns valid actions (empty coordinates)."""
        actions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == ' ':
                    actions.append((r, c))
        return actions

    def transition(self, action: Tuple[int, int], player: str) -> 'TicTacToeState':
        """Transition function: Returns a new state after applying an action."""
        new_board = copy.deepcopy(self.board)
        r, c = action
        new_board[r][c] = player
        next_player = 'O' if player == 'X' else 'X'
        
        next_state = TicTacToeState(self.size, new_board, next_player)
        next_state.history = self.history + [f"Player {player} moved to ({r}, {c})"]
        return next_state

    def is_winner(self, player: str) -> bool:
        """Goal/Reward check: Checks if the specified player has won."""
        # Rows and Columns
        for i in range(self.size):
            if all(self.board[i][j] == player for j in range(self.size)): return True
            if all(self.board[j][i] == player for j in range(self.size)): return True
        # Diagonals
        if all(self.board[i][i] == player for i in range(self.size)): return True
        if all(self.board[i][self.size - 1 - i] == player for i in range(self.size)): return True
        return False

    def is_terminal(self) -> bool:
        """Terminal test."""
        return self.is_winner('X') or self.is_winner('O') or len(self.get_actions()) == 0

    def display(self):
        """Visual representation of the state."""
        print("\n  0   1   2")
        for idx, row in enumerate(self.board):
            print(f"{idx} " + " | ".join(row))
            if idx < self.size - 1:
                print("  " + "-" * (self.size * 4 - 1))
        print()


# =====================================================================
# CO3: CONSTRAINT SATISFACTION PROBLEM (CSP)
# Using CSP concepts to validate move constraints before searching.
# =====================================================================
class MoveValidatorCSP:
    """
    Validates if a proposed move complies with structural game constraints.
    In a true CSP framework, this ensures variables (cells) conform to domains.
    """
    @staticmethod
    def is_valid_move(state: TicTacToeState, row: int, col: int) -> Tuple[bool, str]:
        # Constraint 1: Boundary Constraint
        if not (0 <= row < state.size and 0 <= col < state.size):
            return False, "FAIL: Out of bounds constraint violated. Use indices 0, 1, or 2."
        
        # Constraint 2: Cell Occlusion Constraint
        if state.board[row][col] != ' ':
            return False, f"FAIL: Cell ({row}, {col}) is already occupied by {state.board[row][col]}."
        
        return True, "PASS: All constraints satisfied."


# =====================================================================
# CO2: GRAPH/STATE-SPACE SEARCH HEURISTICS
# Designing evaluation heuristics with time/space consciousness.
# =====================================================================
class BoardHeuristic:
    @staticmethod
    def evaluate(state: TicTacToeState, player: str) -> int:
        """
        An evaluation metric mimicking A*/Greedy heuristic design.
        Scores features based on open line opportunities.
        """
        opponent = 'O' if player == 'X' else 'X'
        score = 0
        
        def score_line(line: List[str]) -> int:
            if opponent in line and player in line:
                return 0  # Blocked line
            if line.count(player) == state.size - 1 and line.count(' ') == 1:
                return 10  # One move away from winning
            if line.count(opponent) == state.size - 1 and line.count(' ') == 1:
                return -10  # Opponent is one move away from winning
            return line.count(player)

        for r in range(state.size):
            score += score_line(state.board[r])
            score += score_line([state.board[c][r] for c in range(state.size)])
        
        score += score_line([state.board[i][i] for i in range(state.size)])
        score += score_line([state.board[i][state.size - 1 - i] for i in range(state.size)])
        
        return score


# =====================================================================
# CO4: ADVERSARIAL DECISION-MAKING AGENT
# Minimax with Alpha-Beta Pruning under bounded computation (depth limit).
# =====================================================================
class AdversarialAgent:
    def __init__(self, player: str, max_depth: int = 4):
        self.player = player
        self.max_depth = max_depth
        self.nodes_expanded = 0  # To empirically track space/time tradeoffs

    def decision_policy(self, state: TicTacToeState) -> Tuple[Tuple[int, int], List[str]]:
        self.nodes_expanded = 0
        best_move = None
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        reasoning_trace = [f"--- AI ({self.player}) Strategic Evaluation ---"]
        
        for action in state.get_actions():
            next_state = state.transition(action, self.player)
            move_val = self._min_value(next_state, 1, alpha, beta)
            reasoning_trace.append(f"Evaluating action {action}: Quality Score = {move_val}")
            
            if move_val > best_val:
                best_val = move_val
                best_move = action
            alpha = max(alpha, best_val)
            
        reasoning_trace.append(f"Decision: Selected {best_move}. Explored {self.nodes_expanded} nodes.")
        return best_move, reasoning_trace

    def _max_value(self, state: TicTacToeState, depth: int, alpha: float, beta: float) -> int:
        self.nodes_expanded += 1
        if state.is_winner(self.player): return 100 - depth
        if state.is_winner('X' if self.player == 'O' else 'O'): return -100 + depth
        if state.is_terminal() or depth >= self.max_depth: 
            return BoardHeuristic.evaluate(state, self.player)

        v = -math.inf
        for action in state.get_actions():
            next_state = state.transition(action, self.player)
            v = max(v, self._min_value(next_state, depth + 1, alpha, beta))
            if v >= beta: return v  # Alpha-beta pruning
            alpha = max(alpha, v)
        return v

    def _min_value(self, state: TicTacToeState, depth: int, alpha: float, beta: float) -> int:
        self.nodes_expanded += 1
        opponent = 'X' if self.player == 'O' else 'O'
        if state.is_winner(opponent): return -100 + depth
        if state.is_winner(self.player): return 100 - depth
        if state.is_terminal() or depth >= self.max_depth: 
            return BoardHeuristic.evaluate(state, self.player)

        v = math.inf
        for action in state.get_actions():
            next_state = state.transition(action, opponent)
            v = min(v, self._max_value(next_state, depth + 1, alpha, beta))
            if v <= alpha: return v  # Alpha-beta pruning
            beta = min(beta, v)
        return v


# =====================================================================
# CO5: REASONING UNDER UNCERTAINTY
# Modeling environmental noise using a simplified Markov Transition model.
# =====================================================================
class UncertainEnvironment:
    def __init__(self, error_probability: float = 0.05):
        self.error_probability = error_probability

    def resolve_action(self, state: TicTacToeState, intended_action: Tuple[int, int]) -> Tuple[Tuple[int, int], str]:
        valid_actions = state.get_actions()
        if random.random() < self.error_probability and len(valid_actions) > 1:
            remaining_actions = [a for a in valid_actions if a != intended_action]
            actual_action = random.choice(remaining_actions)
            return actual_action, f"UNCERTAINTY TRIGGERED: Intended {intended_action}, but slipped to {actual_action}!"
        return intended_action, f"Deterministic Transition: Move executed at {intended_action}."


# =====================================================================
# CO6: INTEGRATED AI REASONING PIPELINE (EXECUTION ENGINE)
# Orchestrates menus, turns, CSP rules, and the final explainable logs.
# =====================================================================
def get_human_move(game_state: TicTacToeState, player_label: str) -> Tuple[int, int]:
    """Helper method to securely get and validate human inputs."""
    while True:
        try:
            user_input = input(f"[{player_label}] Enter your move as 'row col' (e.g., 1 2): ")
            row, col = map(int, user_input.split())
            
            # CSP validation execution (CO3)
            is_valid, msg = MoveValidatorCSP.is_valid_move(game_state, row, col)
            if is_valid:
                return (row, col)
            else:
                print(f"  {msg} Please try again.")
        except (ValueError, IndexError):
            print("  FAIL: Invalid format input. Enter two integers separated by a single space.")

def play_game():
    print("\n" + "="*45)
    print("      ADVANCED AI TIC-TAC-TOE SYSTEM")
    print("="*45)
    print("1. Human vs. Human")
    print("2. Human vs. Computer (AI Agent)")
    
    mode = ""
    while mode not in ['1', '2']:
        mode = input("Select Game Mode (1 or 2): ").strip()

    # Setup pipeline configurations
    game_state = TicTacToeState(size=3)
    environment = UncertainEnvironment(error_probability=0.08)  # 8% chance of unexpected environmental noise (CO5)
    ai_agent = AdversarialAgent(player='O', max_depth=4)        # Minimax agent configuration (CO4)

    while not game_state.is_terminal():
        game_state.display()
        current_p = game_state.current_player
        
        if mode == '1':
            # Human vs Human Mode
            chosen_move = get_human_move(game_state, f"Player {current_p}")
        else:
            # Human vs Computer Mode
            if current_p == 'X':
                chosen_move = get_human_move(game_state, "Human Player X")
            else:
                print("\n🤖 AI Agent is calculation strategy rules...")
                chosen_move, traces = ai_agent.decision_policy(game_state)
                # CO6 Requirement: Output clear step-by-step trace statements
                for trace in traces:
                    print(f"  [Trace] {trace}")

        # Process chosen move through the dynamic system environment module (CO5)
        actual_move, transition_log = environment.resolve_action(game_state, chosen_move)
        if "UNCERTAINTY" in transition_log:
            print(f"\n⚠️ {transition_log}")
            
        game_state = game_state.transition(actual_move, current_p)
        print("\n" + "-"*30)

    # Game Over - Print Results
    game_state.display()
    print("\n=============================================")
    print("                 GAME OVER                   ")
    print("=============================================")
    
    if game_state.is_winner('X'):
        print("🎉 MATCH RESULT: PLAYER X WINS THE GAME!")
    elif game_state.is_winner('O'):
        print("🎉 MATCH RESULT: PLAYER O WINS THE GAME!")
    else:
        print("🤝 MATCH RESULT: IT'S A DRAW/TIE STATE!")
        
    # CO6 Requirement: Print final pipeline audit trace logs
    print("\n--- FINAL EXPLAINABLE REASONING REPORT ---")
    for step in game_state.history:
        print(f"  -> {step}")
    print("=============================================\n")

def main():
    while True:
        play_game()
        rematch = input("Would you like to play another match? (yes/no): ").strip().lower()
        if rematch not in ['y', 'yes']:
            print("\nThank you for exploring the Integrated AI Pipeline system. Goodbye!")
            break

if __name__ == "__main__":
    main()
