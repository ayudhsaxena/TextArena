from typing import Any, Dict, Optional, Tuple
import random
import re
import textarena as ta

import nltk
from nltk.corpus import words
from nltk import pos_tag
import json

class TwentyQuestionsEnv(ta.Env):
    """
    Twenty Questions game environment.
    """

    def __init__(
        self, 
        hardcore: Optional[bool] = False,
    ):
        """
        Initialize the environment.
        
        Args:
            gamemaster_class: The gamemaster class to use
        """
        self.environment_name = "TwentyQuestions"
        self.hardcore = hardcore

        ## intitialise the game state
        self.state = ta.State(
            num_players=1,
            max_turns=21
        )

        # Initialize the gamemaster
        self.gamemaster = ta.agents.OpenRouterAgent(
            model_name="openai/gpt-4o"  # Consider using a larger model if possible
        )
        self.gamemaster_options = ["Yes", "No", "I don't know"]
        self.gamemaster_context = None
        self.gamemaster_history = []

        ## load the word list
        with open("textarena/envs/single_player/TwentyQuestions/twenty_questions_words.json", "r") as f:
            self.word_list = json.load(f)
            
        if self.hardcore:
            self.word_list = self.word_list.get("hardcore")
        else:
            self.word_list = self.word_list.get("basic")

    @property
    def offline_renderer(self):
        pass

    @property
    def terminal_render_keys(self):
        return []
    
    def get_gamemaster_response(self, action: str) -> str:
        """
        Get the gamemaster's response based on the provided action.

        Args:
            action (str): The player's question or statement.

        Returns:
            str: The gamemaster's response.
        """

        # Validate gamemaster state
        if self.gamemaster_context is None:
            raise ValueError("Gamemaster context is not set.")
        if self.gamemaster_history is None:
            raise ValueError("History is not set.")
        if self.gamemaster_options is None:
            raise ValueError("Gamemaster options are not set.")

        # Format available response options
        options = ", ".join(f"'{opt}'" for opt in self.gamemaster_options)

        # Construct conversation history
        history = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.gamemaster_history)

        # Create prompt
        prompt = (
            f"{self.gamemaster_context}\n"
            f"{history}\n\n"
            f"Q: {action}\n"
            f"Options: {options}\n\n"
            "Please respond with the most appropriate option."
        )

        # Get response from the gamemaster agent
        response = self.gamemaster(prompt).strip()

        # Validate response
        if any(option.lower() in response.lower() for option in self.gamemaster_options):
            self.gamemaster_history.append((action, response))  # Store valid responses
        else:
            response = "I'm sorry, I don't understand. Please try asking again."
            self.gamemaster_history.append((action, response))  # Log fallback response

        return response

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

        ## load the game word
        self.game_theme = random.choice(list(self.word_list.keys()))
        self.game_word = random.choice(self.word_list[self.game_theme])

        ## update the gamemaster
        self.gamemaster_context = (
            f"You are the gamemaster for the game of '20 Questions'.\n"
            f"You will provide responses to the players' questions that guides them into guessing the target word: {self.game_word}\n"
        )
        
        ## reset the game state
        return self.state.reset(
            game_state={
                "target_word": self.game_word,
                "rendered_text": self._render_text() ## TODO: Implement this function
            },
            player_prompt_function=self._generate_player_prompt
        )
    
    def _generate_player_prompt(
        self,
        player_id: int,
        game_state: Dict[int, Any]
    ) -> str:
        """
        Generate the initial prompt for a player.
        
        Args:
            player_id: The player's ID (0 or 1).
            
        Returns:
            str: The initial prompt for the player.
        """
        prompt = (
            f"You are Player {player_id}. You are playing 20 Questions ({'Hardcore' if self.hardcore else 'Basic'}).\n"
            f"The gamemaster has chosen an object that can be one or two words. This object is related to {self.game_theme}. You have to guess this object by asking yes-or-no questions.\n"
            "The game will last for a maximum of 20 questions. After 20 questions, the gamemaster will prompt you to make a guess.\n"
            "You may ask your question in any manner, so long they are not wrapped in square brackets.\n"
            "Then, to make your final word guess, ensure that you wrap it with square brackets, e.g. [plane], [diving bell].\n"
            "As you play, the history of your questions and gamemaster's responses will be displayed."
        )

        return prompt
    
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
            player_id: The player's ID.
            action: The action taken by the player.
            
        Returns:
            Tuple: Observations, rewards, truncated, terminated, and info.
        """

        player_id = self.state.current_player_id

        # print("Game observations", self.state.observations)
        
        ## update the observation
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        ## validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z\s]+)\]")  # e.g. [diving bell]
        action_match = action_search_pattern.search(action)

        if not action_match or (action_match and '?' in action):
            ## if the action is not a guess, or if it is a action but contains a question mark, then it is a question
            gamemaster_response = self.get_gamemaster_response(action)
            if self.state.turn == self.state.max_turns-2:
                gamemaster_response += "\nYou have run out of questions. What is your final guess?"
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=gamemaster_response,
                for_logging=True
            )

        else:
            ## if the action is a guess
            action_text = action_match.group(1).lower()
            if self.game_word in action_text:
                self.state.set_winners(
                    player_ids=[player_id],
                    reason=f"Congratulations! Player {player_id} guessed the word."
                )
            else:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid guess. Player {player_id} guessed incorrectly."
                    )

            self.state.game_state["rendered_text"] = self._render_text() ## TODO: Implement this function

        return self.state.step()

    def _generate_gamemaster_response(self, question: str) -> str:
        """
        Generate a response from the gamemaster to a player's question.
        
        Args:
            question: The question asked by the player.
            
        Returns:
            str: The response from the gamemaster.
        """
        responses = self.gamemaster.respond_to_action(question)

        return responses
    
    def _render_text(self) -> str:
        """
        Render the game state.
        
        Returns:
            str: The rendered game state.
        """
        return f"Game word: {self.game_word}"
    
    def _load_word_list(self, word_list: list) -> list:
        """
        Load the word list for the game.

        Args:
            word_list: The word list to load.

        Returns:
            list: The loaded word list.
        """
        # NN: Noun
        return [word for word in word_list if pos_tag([word])[0][1] in ["NN"]]

