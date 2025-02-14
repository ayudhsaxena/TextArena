from typing import Optional, Tuple, List, Dict, Any
import random
import re
import textarena as ta
from collections import deque

class MinesweeperEnv(ta.Env):
    """
    Minesweeper environment.
    """

    def __init__(
        self,
        difficulty: str = "easy",
    ):
        """
        Initialize the Minesweeper environment.
        
        Args:
            difficulty: The difficulty level of the game. Options are "easy", "medium", and "hard".
        """
        self.environment = "Minesweeper"
        self.difficulty = difficulty

        ## set the number of rows, columns, and mines based on the difficulty level
        if self.difficulty == "easy":
            self.rows = 8
            self.cols = 8
            self.num_mines = 10
        elif self.difficulty == "medium":
            self.rows = 10
            self.cols = 10
            self.num_mines = 20
        elif self.difficulty == "hard":
            self.rows = 12
            self.cols = 12
            self.num_mines = 30

        ## initliaze the game state
        self.state = ta.State(
            num_players=1,
            max_turns=100
        )

    @property
    def offline_renderer(self):
        pass

    @property
    def terminal_render_keys(self):
        return ["rendered_board"]
    
    def reset(
        self,
        seed: Optional[int] = None
    ) -> Optional[ta.Observations]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Random seed for the environment.

        Returns:
            Observations: Initial observations for the player.

        """
        ## seed the random number generator
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        ## initialize the game state
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.revealed = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.flags = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.first_move = True # Track if it's the first move to ensure playability

        ## reset the game state
        return self.state.reset(
            game_state={
                "grid": self.grid,
                "rendered_board": self._render_board()
            },
            player_prompt_function=self._generate_player_prompt
        )
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """
        Generate the player prompt.
        
        Args:
            player_id (int): The ID of the player.

        Returns:
            str: The player prompt.
        """

        prompt = (
            f"You are Player {player_id}. You are playing the Minesweeper game.\n"
            "The objective of the game is to reveal all cells that do not contain mines.\n"
            "To make a move, you can either reveal a cell or place a flag on a suspected mine location using one of the following commands:\n"
            "- 'reveal': Reveal the contents of a specific cell.\n"
            "- 'flag': Place or remove a flag on a specific cell to mark it as a potential mine.\n"
            "To submit your move, type the command followed by the row and column in square brackets.\n"
            "For example:\n"
            "- [reveal 3 2] to reveal the cell in Row 3, Column 2.\n"
            "- [flag 5 6] to place or remove a flag on the cell in Row 5, Column 6.\n"
            "On your first move, you will reveal an area around the cell you choose to ensure a safe start.\n"
            "The current board layout is shown below. Cells that are unrevealed are represented by a dot ('.'), revealed numbers show the count of adjacent mines, and flagged cells are marked with an 'F'.\n"
            "Use logic and deduction to avoid revealing cells with mines!\n"
            "Be mindful not to choose revealed or flagged cells.\n"
            "Here is the current board layout:\n"
        )

        prompt += self.state.game_state["rendered_board"]
        return prompt
    
    def _render_board(self) -> str:
        """
        Render the game board.

        Returns:
            str: The rendered game board.
        """
        board_str = "   " + " ".join([str(c).rjust(2) for c in range(self.cols)]) + "\n"
        for r in range(self.rows):
            row_str = f"{r:2} "
            for c in range(self.cols):
                if self.revealed[r][c]:
                    if self.grid[r][c] == -1:
                        row_str += " * "
                    else:
                        row_str += f" {self.grid[r][c]} "
                elif self.flags[r][c]:
                    row_str += " F "
                else:
                    row_str += " . "
            board_str += row_str + "\n"
        return board_str
        
    def step(
        self,
        action: str
    ) -> Tuple[
        Optional[ta.Observations],
        Optional[ta.Rewards],
        bool,
        bool,
        ta.Info
    ]:
        """
        Take a step in the environment.
        
        Args:
            player_id (int): The ID of the player.
            action (str): The action taken by the player.
        
        Returns:
            Observations: The observations for the player.
            Rewards: The rewards for the player.
            bool: Whether the game is truncated.
            bool: Whether the game is terminated.
            Info: Additional information.
        """

        player_id = self.state.current_player_id
        
        ## Update the observation
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        ## Validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\s(\d+)\s(\d+)\]") # e.g. [reveal 3 2]
        match = action_search_pattern.search(action)

        if match is None:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with a valid action and coordinates in square brackets."
            )
        else:
            action, row, col = match.group(1).lower(), int(match.group(2)), int(match.group(3))
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move. The specified row and column coordinates are out of bounds."
                )
            else:
                if action == "reveal":
                    if self.revealed[row][col] or self.flags[row][col]:
                        # print("REVEALED", self.revealed, "FLAGS", self.flags)
                        self.state.set_invalid_move(
                            player_id=player_id,
                            reason=f"Invalid move. The cell at ({row}, {col}) has already been revealed or flagged."
                        )
                    ## Handle the first move
                    if self.first_move:
                        self.clear_all_flags()
                        self.setup_mines(row, col)
                        self.first_move = False
                    
                    queue = deque([(row, col)])  # Start with the initial cell in the queue
                    self.revealed[row][col] = True  # Mark the initial cell as revealed immediately

                    while queue:
                        current_row, current_col = queue.popleft()

                        # Check if it's a mine
                        if self.grid[current_row][current_col] == -1:
                            self.state.set_invalid_move(
                                player_id=player_id,
                                reason=f"Game over! Player {player_id} hit a mine at ({current_row}, {current_col})."
                            )

                        # If the cell has no adjacent mines, add its neighbors to the queue
                        if self.grid[current_row][current_col] == 0:
                            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                                neighbor_row, neighbor_col = current_row + dr, current_col + dc
                                # Only add to the queue if within bounds and not revealed or flagged
                                if 0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols:
                                    if not self.revealed[neighbor_row][neighbor_col] and not self.flags[neighbor_row][neighbor_col]:
                                        self.revealed[neighbor_row][neighbor_col] = True  # Mark as revealed when adding to queue
                                        queue.append((neighbor_row, neighbor_col))

                    self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=f"Game Board:\n{self._render_board()}",
                            for_logging=False
                        )
                    
                elif action == "flag":
                    if not self.revealed[row][col]:
                        self.flags[row][col] = not self.flags[row][col]
                        # print(f"Flag {'placed' if self.flags[row][col] else 'removed'} at ({row}, {col})")

                    self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=f"Game Board:\n{self._render_board()}",
                            for_logging=False
                        )

                else:
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Invalid move format. Player {player_id} did not respond with a valid action in square brackets."
                    )

        ## Update the rendered board
        self.state.game_state["rendered_board"] = self._render_board()

        ## Check if the game is terminated
        if self._is_solved():
            self.state.set_winners(
                player_ids=[player_id],
                reason=f"Congratulations! Player {player_id} has successfully cleared the Minesweeper board."
            )
        
        return self.state.step()
    
    def setup_mines(self, safe_row: int, safe_col: int):
        mines = set()
        while len(mines) < self.num_mines:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            # Avoid placing mines in the safe zone
            if (r, c) not in mines and (r < safe_row - 1 or r > safe_row + 1 or c < safe_col - 1 or c > safe_col + 1):
                mines.add((r, c))
                self.grid[r][c] = -1  # -1 represents a mine
        self.calculate_adjacent_numbers()

    def calculate_adjacent_numbers(self):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1:
                    continue
                mine_count = sum((0 <= r + dr < self.rows and 0 <= c + dc < self.cols and self.grid[r + dr][c + dc] == -1)
                                 for dr, dc in directions)
                self.grid[r][c] = mine_count

    def clear_all_flags(self):
        """Clear all flags on the board."""
        self.flags = [[False for _ in range(self.cols)] for _ in range(self.rows)]

    def _is_solved(self) -> bool:
        """
        Check if the board is in a solved state.

        Returns:
            bool: True if the board is in a solved state, False otherwise.
        """
        return all(
            (self.grid[r][c] == -1 and self.flags[r][c]) or (self.grid[r][c] != -1 and self.revealed[r][c])
            for r in range(self.rows) for c in range(self.cols)
        )
    
    def render(self):
        return self.state.game_state["rendered_board"]