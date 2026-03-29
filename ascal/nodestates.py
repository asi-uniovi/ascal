from enum import Enum
from fcma import Vm

class NodeStates(Enum):
    BOOTING  =  1 # The node is not running, so it is not billed
    BILLED   =  2 # The node is running, so it is billed, but it can not allocate containers yet
    READY    =  3 # The node is ready to allocate containers
    MOVINGC  =  4 # Containers in the node are in the process of being moved to other nodes
    REMOVING =  5 # The node is in the process of being removed
    REMOVED  =  6 # The node has been removed, so it is not billed
    UPGRADING = 7 # The node is in the process of being upgraded

    @staticmethod
    def get_state(node: Vm) -> 'NodeStates':
        """
        Return the current state of the node based on its history.
        """
        assert node is not None, "Invalid node"
        if len(node.history) == 0:
            return NodeStates.BOOTING
        if node.history[0] not in NodeStates:
            return NodeStates.BOOTING
        else:
            return node.history[0]

    @staticmethod
    def set_state(node: Vm, new_state: 'NodeStates'):
        """
        Set the current state of the node based on its history.
        """
        if not node.history:
            node.history.append(new_state)
        else:
            if node.history[0] != new_state:
                node.history.insert(0, new_state)
