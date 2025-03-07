from enum import Enum
from typing import Optional, List
from .state import State

class ActionType(Enum):
    MOVE = "move"
    RECHARGE = "recharge"

class Action:
    def __init__(self, action_type: ActionType, target_idx: int):
        self.action_type = action_type
        self.target_idx = target_idx

    @staticmethod
    def get_valid_actions(state: State) -> List['Action']:
        """Get all valid actions from current state"""
        valid_actions = []
        
        # Add valid move actions to unvisited nodes
        for node in range(state.drp.num_stations, state.drp.num_customers + state.drp.num_stations):
            if state.encoding[node] == 1:  # Node is unvisited
                travel_time = state.drp.get_travel_time(
                    state.instance, 
                    state._to_matrix_index(state.current_node),
                    state._to_matrix_index(node)
                )
                if travel_time <= state.battery_level:
                    valid_actions.append(Action(ActionType.MOVE, node))

        # Add valid recharge actions
        for station in range(state.drp.num_stations):
            travel_time = state.drp.get_travel_time(
                state.instance,
                state._to_matrix_index(state.current_node),
                state._to_matrix_index(station)
            )
            if travel_time <= state.battery_level:
                valid_actions.append(Action(ActionType.RECHARGE, station))

        return valid_actions

    def execute(self, state: State) -> None:
        """Execute action on given state"""
        if self.action_type == ActionType.MOVE:
            state.visit_node(self.target_idx)
        else:  # RECHARGE
            state.recharge(self.target_idx)

    def get_cost(self, state: State) -> float:
        """Get travel time cost of action"""
        return state.drp.get_travel_time(
            state.instance,
            state._to_matrix_index(state.current_node),
            state._to_matrix_index(self.target_idx)
        )

    def __str__(self):
        action_name = "Move to node" if self.action_type == ActionType.MOVE else "Recharge at station"
        return f"{action_name} {self.target_idx}"