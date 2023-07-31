from copy import deepcopy
import numpy as np
import math

from occupancy_map.pose import Pose
from occupancy_map.map_object import AxisAlignedRect, Point
from occupancy_map.abstract import CellState


def make_map(low_corner, high_corner):
    grid = Map(dim=low_corner.shape[0], state=CellState.unknown)
    grid.set_corners(low_corner, high_corner)
    return grid


class Map:
    # ----- Init functions -----
    def __init__(self, dim=None, grid=None, state=CellState._recurse):
        if dim is None and grid is None:
            raise ValueError
        if state is CellState._recurse and grid is None:
            raise ValueError('Specify children or state.')
        if dim is None and state is CellState._recurse:
            raise ValueError

        self.dim = dim
        self.state = state
        self._bounding_box = None
        self.mass = None

        if self.state == CellState._recurse:
            self.grid = grid
            self.shape, dim = self._get_shape(grid, [])
            if not self._check_grid(grid):
                raise ValueError('All lists need to be the same length.')
            if self.dim is None:
                self.dim = dim
        else:
            self.grid = None

    def _get_shape(self, grid, shape, depth=0):
        if isinstance(grid, list):
            shape.append(len(grid))
            return self._get_shape(grid[0], shape, depth + 1)
        else:
            return shape, depth + 1

    def _check_grid(self, grid, depth=0):
        if not isinstance(grid, list):
            return depth == len(self.shape)
        if len(self.shape) <= depth or len(grid) != self.shape[depth]:
            return False

        for row in grid:
            if not self._check_grid(row, depth=depth+1):
                return False

        return True

    # ----- Properties -----
    @property
    def bounding_box(self):
        if self._bounding_box is None:
            raise RuntimeError('Bounding box has not been set.')
        return self._bounding_box

    @property
    def low_corner(self):
        return self.bounding_box.low_corner

    @property
    def high_corner(self):
        return self.bounding_box.high_corner

    def all_corners(self):
        return self._all_corners([])

    def _all_corners(self, corner, depth=0):
        if depth == self.dim:
            return [corner]

        corners = []

        temp = corner.copy()
        temp.append(self.low_corner[depth])
        corners.extend(self._all_corners(temp, depth+1))

        temp = corner.copy()
        temp.append(self.high_corner[depth])
        corners.extend(self._all_corners(temp, depth+1))

        return corners

    # ----- Set corners of cell -----
    def set_corners(self, low_corner, high_corner):
        if (
            self.state is CellState._recurse and
            (low_corner.shape[0] != len(self.shape) or
             high_corner.shape[0] != len(self.shape))
        ):
            raise RuntimeError(
                'Corners need to have %d coordinates.' % len(self.shape)
            )
        if len(low_corner) != len(high_corner) or len(low_corner) != self.dim:
            raise ValueError

        self._bounding_box = AxisAlignedRect(
            high_corner - low_corner, pose=Pose(low_corner)
        )
        self.mass = np.prod(high_corner - low_corner)

        if self.state is CellState._recurse:
            self._set_corners(
                self.grid,
                low_corner,
                high_corner,
                np.zeros(self.dim, dtype=float),
                np.zeros(self.dim, dtype=float)
            )

    def _set_corners(
        self,
        grid,
        low_corner,
        high_corner,
        low_coords,
        high_coords,
        depth=0
    ):
        if depth == self.dim:
            grid.set_corners(low_coords, high_coords)
            return

        step = (high_corner[depth] - low_corner[depth]) / self.shape[depth]
        for i, row in enumerate(grid):
            low_temp = low_coords.copy()
            low_temp[depth] = low_corner[depth] + i * step
            high_temp = high_coords.copy()
            high_temp[depth] = low_corner[depth] + (i + 1) * step
            self._set_corners(
                row,
                low_corner,
                high_corner,
                low_temp,
                high_temp,
                depth + 1
            )

    # ----- Collapse and split cell -----
    def combine(self, state):
        self.grid = None
        self.state = state

    def split(self, *shape):
        if len(shape) != self.dim:
            raise ValueError(
                'Map is %d-dimensional, input has %d dimensions.' %
                (self.dim, len(shape))
            )

        self.grid = self._split(
            Map(dim=self.dim, state=self.state),
            shape
        )

        self.state = CellState._recurse
        self.shape = shape
        self.set_corners(self.low_corner, self.high_corner)

    def _split(self, elm, shape, depth=0):
        if depth == self.dim:
            return elm
        elm = [deepcopy(elm) for _ in range(shape[self.dim - depth - 1])]
        return self._split(elm, shape, depth + 1)

    def collapse(self):
        any_collapsed = False
        if self.state is CellState._recurse:
            any_collapsed, _ = self._collapse()
        return any_collapsed

    def _collapse(self):
        any_collapsed, state = True, self.state

        if self.state is CellState._recurse:
            any_collapsed, state = self._recurse_collapse(self.grid)
            if state is not None:
                self.combine(state)

        return any_collapsed, state

    def _recurse_collapse(self, grid, depth=0):
        if depth == self.dim:
            return grid._collapse()

        any_collapsed = True
        collapsed = True
        state = None
        for row in grid:
            cell_collapsed, temp = self._recurse_collapse(row, depth + 1)
            state = temp if state is None else state

            any_collapsed = any_collapsed or cell_collapsed
            collapsed = collapsed and (state is temp)

        state = state if collapsed else None
        return any_collapsed, state

    # ----- Point checks -----
    def _point_in_cell(self, point):
        inds = self._point_projects_to_cell(point)
        return len(inds) == 0

    def _point_projects_to_cell(self, point):
        # projects to cell, i.e. if some coordinates of the point
        # is removed the point will be in the cell. The point
        # can be projected along some axes to be in the cell
        position = point.pose.position
        return (position < self.low_corner) | (self.high_corner < position)

    def _point_smallest_distance_to_cell(self, point):
        position = point.pose.position
        inds = self._point_projects_to_cell(point)

        if not np.any(inds):
            return 0.0

        corner = np.zeros(self.dim, dtype=float)
        lower = position < self.low_corner

        corner[lower] = self.low_corner[lower]
        corner[~lower] = self.high_corner[~lower]

        return np.sum(np.square(position[inds] - corner[inds]))

    # ----- Fit map to data -----
    def fit_lidar_readings(self, readings):
        if self.dim == 2:
            self._fit_2d_lidar_readings(readings)
        else:
            raise RuntimeError(
                'Cannot fit lidar readings to map which is not '
                'two-dimensional.'
            )

    def _fit_2d_lidar_readings(self, readings):
        raise NotImplementedError

    def fit_point_cloud(self, point_cloud, smallest_mass=0.1):
        is_in_cell = np.full(point_cloud.shape[1], True)

        for i in range(self.dim):
            is_in_cell = is_in_cell & (
                (self.low_corner[i] < point_cloud[i]) &
                (point_cloud[i] < self.high_corner[i])
            )

        if np.any(is_in_cell):
            if self.mass < smallest_mass:
                self.state = CellState.occupied
                return
            point_cloud = point_cloud[:, is_in_cell]
            self.split(*(2 for _ in range(self.dim)))
            self._recurse_point_cloud(self.grid, point_cloud, smallest_mass)
        else:
            self.state = CellState.empty

    def _recurse_point_cloud(self, cells, point_cloud, smallest_mass, depth=0):
        if depth == self.dim:
            cells.fit_point_cloud(point_cloud, smallest_mass)
            return

        for row in cells:
            self._recurse_point_cloud(row, point_cloud, smallest_mass, depth+1)

    # ----- Collision of object with cell state -----
    def object_state(self, map_object, states):
        if not isinstance(states, list):
            states = [states]
        return self._object_state(map_object, states)

    def _object_state(self, map_object, states):
        collides = map_object.collides_with(self.bounding_box)

        if not collides:
            return None

        if self.state is CellState._recurse:
            return self._recurse_object_state(map_object, states, [[]])
        else:
            return self.state in states

    def _recurse_object_state(self, map_object, states, indices, depth=0):
        if depth == self.dim:
            not_none = False
            for index in indices:
                cell = self.grid
                invalid = False

                for i, ind in enumerate(index):
                    if not (0 <= ind < self.shape[i]):
                        invalid = True
                        break
                    cell = cell[ind]

                if not invalid:
                    res = cell._object_state(map_object, states)
                    if res is None:
                        continue
                    not_none = True
                    if res:
                        return True

            return False if not_none else None

        step = (
            (self.high_corner[depth] - self.low_corner[depth]) /
            self.shape[depth]
        )

        lo_index = int(
            (map_object.low_corner[depth] - self.low_corner[depth]) // step
        )
        hi_index = int(
            (map_object.high_corner[depth] - self.low_corner[depth]) // step
        )

        if self.shape[depth] < lo_index:
            return None
        if hi_index < 0:
            return None

        index = []
        for ind in indices:
            for i in range(lo_index, hi_index+1):
                temp = ind.copy()
                temp.append(i)
                index.append(temp)

        return self._recurse_object_state(map_object, states, index, depth + 1)

    # ----- Find smallest distance of point to state -----
    def smallest_distance(self, point, states):
        if not isinstance(point, Point):
            raise ValueError
        if point.abstract:
            raise ValueError

        if not np.all(
            (self.low_corner <= point.pose.position) &
            (point.pose.position <= self.high_corner)
        ):
            raise ValueError('Point is not in map.')

        if not isinstance(states, list):
            states = [states]

        return math.sqrt(self._smallest_distance(point, states))

    def _smallest_distance(self, point, states, dist=float('inf')):
        if self.state is CellState._recurse:
            if self._point_smallest_distance_to_cell(point) < dist:
                return self._recurse_smallest_distance(
                    self.grid,
                    point,
                    states,
                    dist
                )
            else:
                return dist

        if self.state not in states:
            return dist

        temp = self._point_smallest_distance_to_cell(point)
        if temp < dist:
            dist = temp

        return dist

    def _recurse_smallest_distance(self, grid, point, states, dist, depth=0):
        if depth == self.dim:
            return grid._smallest_distance(point, states, dist=dist)

        for row in grid:
            dist = self._recurse_smallest_distance(row, point, states, dist, depth+1)

        return dist
