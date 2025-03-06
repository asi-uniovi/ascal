from enum import Enum
from fcma import Vm

class NodeStates(Enum):
    BOOTING  = 1 # The node is not running, so it is not billed
    BILLED   = 2 # The node is running, so it is billed, but it can not allocate containers yet
    READY    = 3 # The node is ready to allocate containers
    REMOVING = 4 # The node is in the process of being removed
    REMOVED  = 5  # The node has been removed, so it is not billed.

    @staticmethod
    def get_state(node: Vm) -> 'NodeStates':
        if len(node.history) == 0:
            return NodeStates.BOOTING
        if node.history[0] not in [state for state in NodeStates]:
            return NodeStates.BOOTING
        else:
            return node.history[0]

    @staticmethod
    def set_state(node: Vm, new_state: 'NodeStates'):
        if len(node.history) == 0:
            node.history.append(new_state)
        else:
            node.history.insert(0, new_state)
