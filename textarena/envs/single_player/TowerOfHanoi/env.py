from typing import Any, Dict, Optional, Tuple, Union
import copy
import random
import textarena as ta
import re

class TowerOfHanoiEnv(ta.Env):
    """
    Tower of Hanoi game environment.
    """

    def __init__(
        self, 
        difficulty: Optional[str] = "easy",
    ):
        """
        Initialize the envrionment.
        
        Args:
            difficulty: Difficulty of the game.
        """
        self.environment_name = "TowerOfHanoi"
        self.difficulty = difficulty

        if difficulty == "easy":
            self.num_disks = 3
        elif difficulty == "medium":
            self.num_disks = 4
        elif difficulty == "hard":
            self.num_disks = 5

        ## intitialise the game state
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
        seed: Optional[int] = None,
    ) -> Optional[ta.Observations]:
        """
        Reset the environment.

        Args:
            seed: Random seed for the environment.

        Returns:
            Observations: Initial observations.
        """

        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        ## load the game state
        self.towers = self._generate_board()

        ## reset the game state
        return self.state.reset(
            game_state={
                "towers": copy.deepcopy(self.towers),
                "rendered_board": self._render_board()
            },
            player_prompt_function=self._generate_player_prompt
        )
    
    def _generate_player_prompt(
        self, 
        player_id: int,
        game_state: Dict[int, Any]
    ) -> str:
        """
        Generate a player prompt.

        Args:
            player_id: Player ID.
            game_state: Game state.

        Returns:
            str: Player prompt.
        """
        prompt = (
            f"You are Player {player_id}. You are playing Tower of Hanoi ({self.difficulty}).\n"
            f"You have to move the disks from tower A to tower C.\n"
            "To move a disk, type the source tower and the target tower (e.g., '[A C]').\n"
            "Note that you can only move the top disk of a tower, and that a bigger disk cannot be placed on a smaller disk.\n"
            "As you play, the history of your moves will be displayed.\n"
            "Here is the current state of the towers:\n"
        )

        prompt += self._render_board()

        return prompt

    def _generate_board(self):
        """
        Generate the board.
        """
        towers = {
            "A": list(range(self.num_disks, 0, -1)),
            "B": [],
            "C": []
        }

        return towers
    
    def _render_board(self):
        """
        Render the board.
        """
        rendered_board = ""
        for tower, disks in self.towers.items():
            rendered_board += f"{tower}: {disks}\n"

        return rendered_board
    
    def step(
        self,
        action: str,
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
            player_id: The ID of the player.
            action: The action to take. 

        Returns:
            Observations: The observations for the player.
            Rewards: The rewards for the player.
            bool: Whether the episode has ended.
            bool: Whether the episode has been truncated.
            Info: Additional information.
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
        import re

        action_search_pattern = re.compile(r"\[([ABCabc])\s*,?\s*([ABCabc])\]") # e.g. [A, C], [A C], [a c], [a, c]

        matches = action_search_pattern.findall(action)

        if not matches:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with valid '[source] [target]'."
            )
        else:
            for match in matches:
                source, target = match
                source = source.upper()
                target = target.upper()
                if source not in self.towers or target not in self.towers:
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Invalid move. Player {player_id} specified an invalid source or target tower."
                    )
                    break
                elif not self.towers[source]:
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Invalid move. Player {player_id} tried to move a disk from an empty tower."
                    )
                elif self.towers[target] and self.towers[target][-1] < self.towers[source][-1]:
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Invalid move. Player {player_id} tried to place a larger disk on a smaller disk."
                    )
                else:
                    disk = self.towers[source].pop()
                    self.towers[target].append(disk)
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=(f"Player {player_id} moved disk from {source} to {target}. Here is the current state of the towers:\n{self._render_board()}"),
                        for_logging=False
                        )

            ## check if the game is over
            if self.towers["C"] == list(range(self.num_disks, 0, -1)):
                self.state.set_winners(
                    player_ids=[player_id],
                    reason=f"Congratulations! Player {player_id} solved the Tower of Hanoi puzzle."
                )
            
            self.state.game_state["rendered_board"] = self._render_board()

        return self.state.step()
    
    def render(
        self,
        player_id: int,
    ) -> str:
        """
        Render the game state.

        Args:
            player_id: Player ID.

        Returns:
            str: The rendered game state.
        """
        return self.state.game_state["rendered_board"]
    
                    