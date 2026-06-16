# Intelligent_Tic_Tac_Toe_Gaming_Agent
ADVANCED AI TIC-TAC-TOE: INTEGRATED REASONING PIPELINE

An advanced, terminal-based Tic-Tac-Toe application engineered as a complete AI reasoning pipeline in Python. Unlike standard implementations, this project scales beyond simple state tracking to model constraints, evaluate adversarial search depth limits, manage stochastic environments, and provide full human-readable decision traces.

The architecture is explicitly designed to satisfy and demonstrate Course Outcomes CO1 through CO6.

GAMEPLAY FEATURES AND MODES Mode 1: Human vs. Human Local multiplayer module utilizing absolute constraint checking.

Mode 2: Human vs. Computer Play against an advanced AI agent powered by bounded adversarial lookahead search.

Environmental Friction Optional stochastic transitions where input actions can experience subtle environmental noise (cell slippage).

Explainable Diagnostics Prints evaluation traces, node expansion metrics, and a total history log after every match.

Interactive Loops Features an intuitive main menu and a seamless rematch prompt.

COURSE OUTCOME (CO) MAPPING This project serves as a practical capstone demonstrating the following core competencies:

CO1: Problem and State Space Formulation

Implementation: TicTacToeState Class

Details: Formulates the real-world board layout using nested Python lists. It defines mathematical state spaces, trackable action arrays via get_actions(), strict transition mechanics through transition(), and terminal goal validations using is_winner().

CO2: Graph/State-Space Search Heuristics

Implementation: BoardHeuristic Class

Details: Implements a line-scoring heuristic evaluation metric reminiscent of A-star or Greedy search components. It judges potential paths based on open-line win opportunities and counts expanded nodes to analyze empirical execution trade-offs.

CO3: Constraint Satisfaction Problems (CSP)

Implementation: MoveValidatorCSP Class

Details: Formulates move validation under a strict CSP framework. The coordinates are treated as variables mapped against finite domains. It strictly enforces boundary limitations and cell occlusion rules before modifying variables.

CO4: Adversarial Decision-Making Agent

Implementation: AdversarialAgent Class

Details: Deploys a complete Minimax lookahead agent optimized with Alpha-Beta Pruning. It applies a fixed depth limit parameter to successfully simulate tactical decision-making under bounded computational resources.

CO5: Reasoning Under Uncertainty

Implementation: UncertainEnvironment Class

Details: Simulates real-world environmental noise. Moves do not always execute deterministically; instead, the system utilizes a probability threshold model to dynamically shift intended actions to adjacent valid cells, mimicking Markov-like transitional slips.

CO6: Integrated AI Reasoning Pipeline

Implementation: play_game() Execution Engine

Details: Seamlessly orchestrates every component—State representation, CSP constraints, Adversarial lookahead, and Environmental Uncertainty—into an integrated loop. It outputs clear, explainable traces demonstrating the step-by-step logic of the system.
