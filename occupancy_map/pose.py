from copy import deepcopy
import numpy as np
from scipy.linalg import expm, logm


class WorldCoordinates:
    def align_to_pose(self, pose):
        raise NotImplementedError

    def translate_by(self, translation):
        raise NotImplementedError

    def rotate_by(self, rotation):
        raise NotImplementedError


class Position(WorldCoordinates):
    def __init__(self, position):
        self.position = np.array(position)
        self.dim = len(position)

    def __str__(self):
        string = 'Position['
        for i in range(self.position.shape[0]):
            if i > 0:
                string += ', '
            string += 'x%d=%.3f' % (i + 1, self.position[i])
        string += ']'
        return string

    def align_to_pose(self, pose):
        self.rotate_by(pose._orientation)
        self.translate_by(pose._position)
        return self

    def translate_by(self, translation):
        if translation.dim != self.dim:
            raise ValueError
        self.position += translation.position
        return self

    def rotate_by(self, rotation):
        self.position = rotation.rot_mat @ self.position
        return self


class Orientation(WorldCoordinates):
    def __init__(self, orientation):
        self.dim = 1 + int((1 + np.sqrt(1 + 4 * len(orientation))) // 2)
        self.orientation = np.array(orientation)

        if (self.dim * (self.dim - 1)) // 2 != len(orientation):
            raise ValueError

    def __str__(self):
        string = 'Orientation['
        for i in range(self._orientation.shape[0]):
            if i > 0:
                string += ', '
            string += 'q%d=%.3f' % (i + 1, self._orientation[i])
        string += ']'
        return string

    def rotate_by(self, rotation):
        if rotation.dim != self.dim:
            raise ValueError
        self.rot_mat = self.rot_mat @ rotation.rot_mat
        return self

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation

        mat = np.zeros((self.dim, self.dim), dtype=float)
        i, j = np.triu_indices(n=self.dim, k=1)
        mat[i, j] = self._orientation

        mat -= mat.T
        self._rot_mat = expm(mat)

    @property
    def rot_mat(self):
        return self._rot_mat

    @rot_mat.setter
    def rot_mat(self, rot_mat):
        self._rot_mat = rot_mat

        i, j = np.triu_indices(n=self.dim, k=1)
        mat = logm(self._rot_mat)
        self._orientation = mat[i, j]


class Pose(WorldCoordinates):
    def __init__(self, position, orientation=None):
        if isinstance(position, Position):
            self._position = position
        else:
            self._position = Position(position)
        if isinstance(orientation, Orientation):
            self._orientation = orientation
        else:
            if orientation is None:
                dim = self._position.dim
                n = ((dim - 1) * dim) // 2
                orientation = np.zeros(n, dtype=float)
            self._orientation = Orientation(orientation)

        if self._position.dim != self._orientation.dim:
            raise ValueError
        self.dim = self._position.dim

    def __str__(self):
        return 'Pose[%s, %s]' % (self._position, self._orientation)

    @property
    def position(self):
        return self._position.position

    @property
    def orientation(self):
        return self._orientation.orientation

    @property
    def rot_mat(self):
        return self._orientation.rot_mat

    def copy(self):
        return deepcopy(self)

    def align_to_pose(self, pose):
        self._position.align_to_pose(pose)
        self._orientation.rotate_by(pose._orientation)
        return self

    def translate_by(self, translation):
        self.position.translate_by(translation)
        return self

    def rotate_by(self, rotation):
        self.orientation.rotate_by(rotation)
        return self


if __name__ == '__main__':
    root = Pose(Position([1.0, 0.0, 0.0]), Orientation([0.0, -np.pi/2, 0.0]))
    fr1 = Pose(Position([1.0, 0.0, 0.0]), Orientation([0.0, np.pi/2, 0.0]))
    root.align_to_pose(fr1)

    print(root)
    print(fr1)
