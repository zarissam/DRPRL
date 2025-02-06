from typing import Optional, List
import numpy as np
from environment.state import State
from environment.action import Action
from environment.cost import Cost
from learning.value_network import ValueNetwork

class Policy:
    def __init__(self, value_network: ValueNetwork, cost_function: Cost):
        self.value_network = value_network
        self.cost_function = cost_function

    def select_action(self, state: State) -> Action:
        """
        Implements π(s) = argmin_a [c(s,a) + V(T(s,a))]
        """
        valid_actions = Action.get_valid_actions(state)
        if not valid_actions:
            raise ValueError("No valid actions available")

        best_action = None
        min_total_cost = float('inf')

        for action in valid_actions:
            # Calculate immediate cost c(s,a)
            immediate_cost = self.cost_function(state, action)

            # Create next state T(s,a)
            next_state = state.__class__(
                state.drp,
                state.instance,
                state.current_node
            )
            action.execute(next_state)

            # Calculate future cost V(T(s,a))
            future_cost = self.value_network.predict(next_state)

            # Total cost
            total_cost = immediate_cost + future_cost

            # Update best action
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_action = action

        return best_action

    def get_action_values(self, state: State) -> List[tuple]:
        """Returns list of (action, value) pairs for all valid actions"""
        valid_actions = Action.get_valid_actions(state)
        action_values = []

        for action in valid_actions:
            immediate_cost = self.cost_function(state, action)
            
            next_state = state.__class__(
                state.drp,
                state.instance,
                state.current_node
            )
            action.execute(next_state)
            
            future_cost = self.value_network.predict(next_state)
            total_cost = immediate_cost + future_cost
            
            action_values.append((action, total_cost))

        return sorted(action_values, key=lambda x: x[1])