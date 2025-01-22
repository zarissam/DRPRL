from typing import Optional
from .state import State
from .action import Action, ActionType

class Cost:
    def __init__(self):
        """Initialize cost function parameters"""
        pass

    def calculate(self, state: State, action: Action) -> float:
        """
        Calculate cost c(s,a) = t_ij (battery swap time t_r = 0)
        Args:
            state: Current state
            action: Action taken
        Returns:
            Total cost (just travel time since battery swap is instant)
        """
        return state.drp.get_travel_time(
            state.instance,
            state._to_matrix_index(state.current_node),
            state._to_matrix_index(action.target_idx)
        )

    def __call__(self, state: State, action: Action) -> float:
        """Make cost object callable"""
        return self.calculate(state, action)