""" Register all environments """

from textarena.envs.registration import (
    make,
    register,
) 

from textarena.game_makers import GPTJudgeVote

# Register Game sets
register(
    id="BalancedSubset-v0",
    entry_point="textarena.envs.two_player.Subsets.env:SubsetEnv",
    env_ids=[
        "DontSayIt-v0",
        "Negotiation-v0",
        "LiarsDice-v0",
        "Chess-v0",
        "TruthAndDeception-v0",
        "SpellingBee-v0",
        "Poker-v0",
        "Stratego-v0",
        "Tak-v0",
        "UltimateTicTacToe-v0",
    ],
    max_num_characters=1_000
)

# Register Individual Games
register(
    id="DontSayIt-v0",
    entry_point="textarena.envs.two_player.DontSayIt.env:DontSayItEnv",
    hardcore=False,
    max_turns=30,
)
register(
    id="DontSayIt-v0-hardcore",
    entry_point="textarena.envs.two_player.DontSayIt.env:DontSayItEnv",
    hardcore=True,
    max_turns=30,
)
register(
    id="DontSayIt-v0-unlimited",
    entry_point="textarena.envs.two_player.DontSayIt.env:DontSayItEnv",
    hardcore=False,
    max_turns=None,
)

register(
    id="Negotiation-v0",
    entry_point="textarena.envs.two_player.Negotiation.env:NegotiationEnv",
    max_turns=10,
)
register(
    id="Negotiation-v0-short",
    entry_point="textarena.envs.two_player.Negotiation.env:NegotiationEnv",
    max_turns=6,
)
register(
    id="Negotiation-v0-long",
    entry_point="textarena.envs.two_player.Negotiation.env:NegotiationEnv",
    max_turns=30,
)

register(
    id="IteratedPrisonersDilemma-v0",
    entry_point="textarena.envs.two_player.IteratedPrisonersDilemma.env:IteratedPrisonersDilemmaEnv",
    num_rounds=10,
    communication_turns=3,
    cooperate_reward=3,
    defect_reward=5,
    sucker_reward=0,
    mutual_defect_reward=1,
)

register(
    id="Chess-v0",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=False,
    max_turns=30,
    show_valid=True,
)
register(
    id="Chess-v0-open",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=True,
    max_turns=30,
    show_valid=False,
)
register(
    id="Chess-v0-long",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=False,
    max_turns=50,
    show_valid=True,
)
register(
    id="Chess-v0-blind",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=False,
    max_turns=50,
    show_valid=False,
)

register(
    id="TruthAndDeception-v0",
    entry_point="textarena.envs.two_player.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=6,
)
register(
    id="TruthAndDeception-v0-long",
    entry_point="textarena.envs.two_player.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=12,
)
register(
    id="TruthAndDeception-v0-super-long",
    entry_point="textarena.envs.two_player.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=50,
)

register(
    id="SpellingBee-v0",
    entry_point="textarena.envs.two_player.SpellingBee.env:SpellingBeeEnv",
    num_letters=6,
)
register(
    id="SpellingBee-v0-small",
    entry_point="textarena.envs.two_player.SpellingBee.env:SpellingBeeEnv",
    num_letters=4,
)
register(
    id="SpellingBee-v0-large",
    entry_point="textarena.envs.two_player.SpellingBee.env:SpellingBeeEnv",
    num_letters=10,
)

register(
    id="Poker-v0",
    entry_point="textarena.envs.two_player.Poker.env:PokerEnv",
    num_rounds=5,
    starting_chips=1_000,
    small_blind=10,
    big_blind=20
)

register(
    id="Poker-v0-long",
    entry_point="textarena.envs.two_player.Poker.env:PokerEnv",
    num_rounds=15,
    starting_chips=1_000,
    small_blind=10,
    big_blind=20
)


register(
    id="Poker-v0-super-long",
    entry_point="textarena.envs.two_player.Poker.env:PokerEnv",
    num_rounds=50,
    starting_chips=1_000,
    small_blind=10,
    big_blind=20
)


register(
    id="Stratego-v0",
    entry_point="textarena.envs.two_player.Stratego.env:StrategoEnv"
)

register(
    id="Tak-v0-easy",
    entry_point="textarena.envs.two_player.Tak.env:TakEnv"
)

register(
    id="Tak-v0",
    entry_point="textarena.envs.two_player.Tak.env:TakEnv"
)

register(
    id="Tak-v0-hard",
    entry_point="textarena.envs.two_player.Tak.env:TakEnv"
)

register(
    id="UltimateTicTacToe-v0",
    entry_point="textarena.envs.two_player.UltimateTicTacToe.env:UltimateTicTacToeEnv"
)

register(
    id="LiarsDice-v0",
    entry_point="textarena.envs.two_player.LiarsDice.env:LiarsDiceEnv",
    num_dice=5,
)

register(
    id="LiarsDice-v0-large",
    entry_point="textarena.envs.two_player.LiarsDice.env:LiarsDiceEnv",
    num_dice=12,
)