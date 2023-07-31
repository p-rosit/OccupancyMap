import numpy as np
from occupancy_map.pose import Pose
from occupancy_map.map_object import AxisAlignedRect


class CellState:
    empty = 0
    occupied = 1
    unknown = 2
    _recurse = 3

    @staticmethod
    def col(state, transparent=False):
        if transparent:
            return CellState._transparent_col(state)

        if state is CellState.empty:
            col = (0.9, 0.9, 0.9)
        elif state is CellState.occupied:
            col = (0.1, 0.1, 0.1)
        elif state is CellState.unknown:
            col = (0.6, 0.6, 0.1)
        else:
            raise ValueError

        return col

    def _transparent_col(state):
        col = CellState.col(state)
        if state is CellState.empty:
            col = (*col, 0.0)
        elif state is CellState.occupied:
            col = (*col, 0.2)
        elif state is CellState.unknown:
            col = (*col, 0.2)
        else:
            raise ValueError

        return col


class AbstractMap:
    def __init__(self, low_corner, high_corner, state):
        if isinstance(low_corner, list):
            low_corner = np.array(low_corner, dtype=float)
        if isinstance(high_corner, list):
            high_corner = np.array(high_corner, dtype=float)
        if low_corner.shape[0] != high_corner.shape[0]:
            raise ValueError

        self.bounding_box = AxisAlignedRect(
            high_corner - low_corner,
            pose=Pose(low_corner)
        )
        self.mass = np.prod(high_corner - low_corner)

        self.dim = low_corner.shape[0]
        self.background = state
        self.states = [state]

        self.shape = None
        self.contents = dict()

        self.mass = None

    # ----- Collapse and split cell -----
