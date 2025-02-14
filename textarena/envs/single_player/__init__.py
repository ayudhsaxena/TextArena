""" Register all environments """

from textarena.envs.registration import register


######################################### Single Player Games #########################################
register(
    id="Crosswords-v0",
    entry_point="textarena.envs.single_player.Crosswords.env:CrosswordsEnv",
    hardcore=False,
    max_turns=30,
    num_words=3,
)

register(
    id="Crosswords-v0-hardcore",
    entry_point="textarena.envs.single_player.Crosswords.env:CrosswordsEnv",
    hardcore=True,
    max_turns=30,
    num_words=3,
)

register(
    id="Sudoku-v0-easy",
    entry_point="textarena.envs.single_player.Sudoku.env:SudokuEnv",
    difficulty="easy",
    max_turns=100,
)

register(
    id="Sudoku-v0-medium",
    entry_point="textarena.envs.single_player.Sudoku.env:SudokuEnv",
    difficulty="medium",
    max_turns=100,
)

register(
    id="Sudoku-v0-hard",
    entry_point="textarena.envs.single_player.Sudoku.env:SudokuEnv",
    difficulty="hard",
    max_turns=100,
)

register(
    id="Hangman-v0",
    entry_point="textarena.envs.single_player.Hangman.env:HangmanEnv",
    hardcore=False,
)

register(
    id="Hangman-v0-hardcore",
    entry_point="textarena.envs.single_player.Hangman.env:HangmanEnv",
    hardcore=True,
)

register(
    id="GuessTheNumber-v0",
    entry_point="textarena.envs.single_player.GuessTheNumber.env:GuessTheNumberEnv",
    hardcore=False,
)

register(
    id="GuessTheNumber-v0-hardcore",
    entry_point="textarena.envs.single_player.GuessTheNumber.env:GuessTheNumberEnv",
    hardcore=True,
)

register(
    id="WordSearch-v0",
    entry_point="textarena.envs.single_player.WordSearch.env:WordSearchEnv",
    hardcore=False,
)

register(
    id="WordSearch-v0-hardcore",
    entry_point="textarena.envs.single_player.WordSearch.env:WordSearchEnv",
    hardcore=True,
)

register(
    id="WordLadder-v0-easy",
    entry_point="textarena.envs.single_player.WordLadder.env:WordLadderEnv",
    difficulty="easy",
)

register(
    id="WordLadder-v0-medium",
    entry_point="textarena.envs.single_player.WordLadder.env:WordLadderEnv",
    difficulty="medium",
)

register(
    id="WordLadder-v0-hard",
    entry_point="textarena.envs.single_player.WordLadder.env:WordLadderEnv",
    difficulty="hard",
)

register(
    id="FifteenPuzzle-v0",
    entry_point="textarena.envs.single_player.FifteenPuzzle.env:FifteenPuzzleEnv",
)

register(
    id="LogicPuzzle-v0",
    entry_point="textarena.envs.single_player.LogicPuzzle.env:LogicPuzzleEnv",
    difficulty="easy",
)

register(
    id="LogicPuzzle-v0-hard",
    entry_point="textarena.envs.single_player.LogicPuzzle.env:LogicPuzzleEnv",
    difficulty="hard",
)

register(
    id="TwentyQuestions-v0",
    entry_point="textarena.envs.single_player.TwentyQuestions.env:TwentyQuestionsEnv",
    hardcore=False,
)

register(
    id="TwentyQuestions-v0-hardcore",
    entry_point="textarena.envs.single_player.TwentyQuestions.env:TwentyQuestionsEnv",
    hardcore=True,
)

register(
    id="TowerOfHanoi-v0-easy",
    entry_point="textarena.envs.single_player.TowerOfHanoi.env:TowerOfHanoiEnv",
    difficulty="easy"
)

register(
    id="TowerOfHanoi-v0-medium",
    entry_point="textarena.envs.single_player.TowerOfHanoi.env:TowerOfHanoiEnv",
    difficulty="medium"
)

register(
    id="TowerOfHanoi-v0-hard",
    entry_point="textarena.envs.single_player.TowerOfHanoi.env:TowerOfHanoiEnv",
    difficulty="hard"
)

register(
    id="Minesweeper-v0-easy",
    entry_point="textarena.envs.single_player.Minesweeper.env:MinesweeperEnv",
    difficulty="easy",
)

register(
    id="Minesweeper-v0-medium",
    entry_point="textarena.envs.single_player.Minesweeper.env:MinesweeperEnv",
    difficulty="medium",
)

register(
    id="Minesweeper-v0-hard",
    entry_point="textarena.envs.single_player.Minesweeper.env:MinesweeperEnv",
    difficulty="hard",
)

register(
    id="GuessWho-v0",
    entry_point="textarena.envs.single_player.GuessWho.env:GuessWhoEnv",
)

register(
    id="Chess-v0-singleplayer",
    entry_point="textarena.envs.single_player.Chess.env:ChessEnv",
)

register(
    id="ConnectFour-v0-singleplayer",
    entry_point="textarena.envs.single_player.ConnectFour.env:ConnectFourEnv",
)
