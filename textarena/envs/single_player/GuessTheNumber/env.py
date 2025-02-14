from typing import Any, Dict, Optional, Tuple, Union
import copy
import random
import textarena as ta
import re

class GuessTheNumberEnv(ta.Env):
    """
    Guess the number game environment.
    """

    def __init__(
        self,
        hardcore: Optional[bool] = False,
    ):
        """
        Initialize the environment.

        Args:
            hardcore: Whether to play in hardcore mode.
        """
        self.environment_name = "GuessTheNumber"
        self.hardcore = hardcore

        if not hardcore:
            self.min_number = 1
            self.max_number = 20
        else:
            self.min_number = 1
            self.max_number = 100

        ## intitialise the game state
        self.state = ta.State(
            num_players=1,
            max_turns=10
        )

    @property
    def offline_renderer(self):
        pass

    @property
    def terminal_render_keys(self):
        return ["rendered_text"]
    

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

        ## load the game number
        self.game_number = random.randint(self.min_number, self.max_number)
        self.guessed_numbers = set()

        ## reset the game state
        return self.state.reset(
            game_state={
                "game_number": self.game_number,
                "rendered_text": "Guess the number between {} and {}.".format(self.min_number, self.max_number),
            },
            player_prompt_function=self._generate_player_prompt
        )
    
    def _generate_player_prompt(
        self, 
        player_id: int,
        game_state: Dict[int, Any]
    ) -> str:
        """
        Generate the player prompt.
        
        Args:
            player_id: The player id.
            
        Returns:
            str: The player prompt.
        """
        prompt = (
            f"You are Player {player_id}. You are playing Guess The Number ({'Hardcore' if self.hardcore else 'Basic'}).\n"
            f"You have to guess the number between {self.min_number} and {self.max_number}.\n"
            "As you enter your guess, the game will provide you with hints such as 'higher' or 'lower'.\n"
            "You may provide your response in any manner. Only the number that is wrapped in square brackets will be considered as your guess. For example, [5].\n"
            "As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.\n"
            "Enter your guess."
        )

        return prompt
    
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
            player_id: The ID of the player.
            action: The action taken by the player.

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
        action_search_pattern = re.compile(r"\[(\d+)\]") # e.g. [5]
        match = action_search_pattern.search(action)

        if not match:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with valid '[number]'."
            )
        else:
            player_guess = int(match.group(1))
            if player_guess < self.min_number or player_guess > self.max_number:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move. Player {player_id} guessed a number outside the range specified."
                )
            elif player_guess in self.guessed_numbers:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move. Player {player_id} has already guessed the number."
                )
            else:
                self.guessed_numbers.add(player_guess)
                if player_guess == self.game_number:
                    self.state.set_winners(
                        player_ids=[player_id],
                        reason=f"Congratulations! Player {player_id} guessed the correct number."
                    )
                else:
                    if player_guess < self.game_number:
                        hint = "lower"
                    else:
                        hint = "higher"
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=f"Your guess of {player_guess} is {hint}.",
                        for_logging=False
                    )

            self.state.game_state["rendered_text"] = f"Player {player_id} guessed {player_guess}."

        return self.state.step()
