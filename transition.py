from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from fcma import System, Allocation, App, ContainerClass, InstanceClass
from timedops import TimedOps
from recycling import Recycling

class Command(Enum):
    CREATE_NODE = 1
    REMOVE_NODE = 2
    CREATE_CONTAINER = 3
    REMOVE_CONTAINER = 4

@dataclass
class TransitionStatistics:
    transition_times: dict[App, int]
    recycling_level: dict[InstanceClass, float]

class Transition:
    def __init__(self, timing_args: TimedOps.TimingArgs, system: System):
        self. timing_args = timing_args
        self.system = system
        self.initial_alloc = None
        self.final_alloc = None
        self.initial_alloc = None
        self.final_alloc = None
        self.recycling = None
        self.time_commands: list[tuple[int, Command]] = []
        self.current_alloc: Allocation = None

    def allocate_attempt1(self, cc: ContainerClass, replicas: int):
        pass

    def allocate_attempt2(self, cc: ContainerClass, replicas: int):
        pass

    def allocate_attempt3(self, cc: ContainerClass, replicas: int):
        pass

    def calculate(self, initial_alloc: Allocation, final_alloc: Allocation, apps: tuple[App]|None=None):
        if initial_alloc != self.final_alloc and final_alloc != self.final_alloc:
            self.initial_alloc = initial_alloc
            self.final_alloc = final_alloc
            self.recycling = Recycling(self.initial_alloc, final_alloc)
            self.current_alloc = deepcopy(self.initial_alloc)
        pass

    def get_transition_times(self) -> dict[App, int]:
        pass


