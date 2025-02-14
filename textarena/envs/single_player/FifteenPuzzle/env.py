from typing import Any, Dict, List, Tuple, Optional, Union
import random
import textarena as ta
import re

class FifteenPuzzleEnv(ta.Env):
    """
    Fifteen Puzzle environment.
    """

    def __init__(
        self,
    ):
        """
        Initialize the Fifteen Puzzle environment.
        """

        super().__init__()
        self.environment_name = "FifteenPuzzle"

        ## initialize the game state
        self.state = ta.State(
            num_players=1,
            max_turns=50
        )

    @property
    def offline_renderer(self):
        from textarena.envs.single_player.FifteenPuzzle.render.renderer import FifteenPuzzleRenderer
        return FifteenPuzzleRenderer

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
        self.board = self._generate_board()
        
        ## reset the game state
        return self.state.reset(
            game_state={
                "board": self.board,
                "rendered_board": self._render_board(self.board)
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
            f"You are Player {player_id}. You are playing the 15-Puzzle game.\n"
            "The objective of the game is to arrange the numbered tiles in ascending order from 1 to 15, with the empty space located in the bottom-right corner.\n"
            "To make a move, you can slide a tile into the empty space (represented by a double underscore, e.g. __) by using one of the following commands:\n"
            "- 'up': Move the tile below the empty space up.\n"
            "- 'down': Move the tile above the empty space down.\n"
            "- 'left': Move the tile to the right of the empty space left.\n"
            "- 'right': Move the tile to the left of the empty space right.\n"
            "To submit your move, type the direction (e.g., 'up', 'down', 'left', or 'right') in square brackets, e.g. [up].\n"
            "The current board layout is shown below. Use the information to solve the puzzle.\n"
        )

        prompt += self.state.game_state["rendered_board"]
        return prompt
    
    def _generate_board(self):
        """
        Generate a shuffled board configuration.

        Returns:
            List[List[Optional[int]]]: A 4x4 grid representing the board configuration

        """
        tiles = list(range(1, 16)) + [None]
        random.shuffle(tiles)
        return [tiles[i:i + 4] for i in range(0, 16, 4)] # e.g. [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, None]]
    
    def _render_board(self, board):
        """
        Render the current board layout.

        Args:
            board (List[List[Optional[int]]]): The 4x4 grid representing the board configuration.

        Returns:
            str: The rendered board layout.

        """
        rendered_board = ""
        for row in board:
            rendered_board += ' '.join(['__' if x is None else f"{x:2}" for x in row]) + "\n"
        return rendered_board
    
    def step(
        self,
        action: str
    ) -> Tuple[
        Optional[ta.Observations], # observations
        Optional[ta.Rewards], # reward
        bool, # truncated
        bool, # terminated
        ta.Info # info
    ]:
        
        """
        Process the player's action and update the environment state.

        Args:
            player_id (int): The ID of the player making the move.
            action (str): The action taken by the player.

        Returns:
            Observations: Observations for the player after the action.
            Rewards: Rewards for the player after the action.
            bool: Whether the game was truncated.
            bool: Whether the game is terminated.
            Info: Additional information about the game state

        """

        player_id = self.state.current_player_id

        ## update the observation
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        ## validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\]") # e.g. [up]
        match = action_search_pattern.search(action)

        if match is None:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with a valid direction in square brackets."
            )
        else:
            direction = match.group(1)
            if not self._move(direction):
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move. The tile cannot be moved in the specified direction."
                )
            else:
                ## update the rendered board
                self.state.game_state["rendered_board"] = self._render_board(self.board)
                self.state.add_observation(
                    from_id=-1,
                    to_id=player_id,
                    message=f"Game Board:\n{self._render_board(self.board)}",
                    for_logging=False
                )
            
        ## check if the puzzle is solved
        if self._is_solved():
            self.state.set_winners(
                player_ids=[player_id],
                reason=f"Congratulations! Player {player_id} have successfully solved the 15-Puzzle."
            )
        
        return self.state.step()
    
    def _is_solved(self) -> bool:
        """
        Check if the board is in a solved state.

        Returns:
            bool: True if the board is in a solved state, False otherwise.

        """
        correct_tiles = list(range(1, 16)) + [None]
        current_tiles = [tile for row in self.board for tile in row]
        return current_tiles == correct_tiles

    def _move(self, direction: str) -> bool:
        """
        Move a tile into the empty space if the direction is valid.

        Args:
            direction (str): Direction to move, one of 'up', 'down', 'left', 'right'.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        empty_row, empty_col = self._get_empty_position()
        target_row, target_col = empty_row, empty_col

        if direction == 'up' and empty_row < 3:
            target_row += 1
        elif direction == 'down' and empty_row > 0:
            target_row -= 1
        elif direction == 'left' and empty_col < 3:
            target_col += 1
        elif direction == 'right' and empty_col > 0:
            target_col -= 1
        else:
            ## invalid move
            return False

        ## swap the target tile with the empty tile
        self.board[empty_row][empty_col], self.board[target_row][target_col] = (
            self.board[target_row][target_col],
            self.board[empty_row][empty_col],
        )
        return True
    
    def _get_empty_position(self):
        """
        Return the current position of the empty tile.

        Returns:
            Tuple[int, int]: The row and column indices of the empty tile.
        """
        for r in range(4):
            for c in range(4):
                if self.board[r][c] is None:
                    return r, c
                