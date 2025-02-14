""" Root __init__ of textarena """

from textarena.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    RenderWrapper,
    ActionWrapper,
    Agent,
    AgentWrapper,
    State,
    Message,
    Observations,
    Rewards,
    Info,
    GAME_ID,
)
from textarena.envs.registration import (
    make,
    register,
    pprint_registry,
    pprint_registry_detailed,
    find_highest_version,
    check_env_exists,
)

# from textarena.api import make_online #, SyncOnlineEnv


from textarena import wrappers, agents



__all__ = [
    # core
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "RenderWrapper",
    "ActionWrapper",
    "AgentWrapper",

    # registration
    "make",
    "register",
    "pprint_registry",
    "pprint_registry_detailed",
    "find_highest_version",
    "check_env_exists",
    
    # module folders
    "envs",
    "utils",
    "wrappers",
    
    # play online
    "make_online",
]
__version__ = "0.5.0"
