from copy import deepcopy
import numpy as np
from occupancy_map.pose import Pose, Position


class MapObject:
    def __init__(self, dim=None, pose=None, position=None):
        self.pose = None
        self.low_corner = None
        self.high_corner = None

        if dim is None and pose is None and position is None:
            raise ValueError
        if dim is None and pose is not None and position is not None:
            raise ValueError
        if dim is None:
            self.dim = position.dim if pose is None else pose.dim
        else:
            self.dim = dim

        self.abstract = pose is None and position is None
        if not self.abstract:
            if pose is None:
                pose = Pose(position)
            self.at_pose(pose, in_place=True)

    def __str__(self):
        info_str = self._info_str()

        string = '%s[' % self.__class__.__name__
        string += '' if info_str == '' else 'info=[%s]' % info_str
        if self.abstract:
            return 'Abstract_' + string + ']'
        elif info_str:
            string += ', '

        string += '%s, %s]' % (self.pose._position, self.pose._orientation)
        return string

    def _info_str(self):
        raise NotImplementedError

    def copy(self):
        return deepcopy(self)

    def at_pose(self, pose, in_place=True):
        if pose.dim != self.dim:
            raise ValueError
        pose = pose.copy()

        if in_place:
            self.abstract = False
            self.pose = pose
        return self._at_pose(pose, in_place)

    def _at_pose(self, pose, in_place):
        raise NotImplementedError

    def collides_with(self, obj):
        collision_with = None
        self._started = False

        if isinstance(obj, AxisAlignedRect):
            collision_with = self.collides_with_rect
        elif isinstance(obj, Point):
            collision_with = self.collides_with_point
        elif isinstance(obj, Sphere):
            collision_with = self.collides_with_sphere
        elif isinstance(obj, ConvexMesh):
            collision_with = self.collides_with_convex_mesh
        elif isinstance(obj, Polygon):
            collision_with = self.collides_with_polygon
        else:
            raise NotImplementedError(
                'Collision between objects of type "%s" and '
                '"%s" has not been implemented.' % (
                    self.__class__.__name__, obj.__class__.__name__
                )
            )

        return collision_with(obj)

    def collides_with_rect(self, rect):
        if self.abstract or rect.abstract:
            raise ValueError
        if self.dim != rect.dim:
            raise ValueError

        if np.any(
            (self.high_corner < rect.low_corner) |
            (rect.high_corner < self.low_corner)
        ):
            return False

        # Bounding box collision succeeded
        return self._collides_with_rect(rect)

    def _collides_with_rect(self, rect):
        if self._started:
            raise NotImplementedError
        self._started = True
        return rect.collides_with(self)


class AxisAlignedRect(MapObject):
    def __init__(self, corner, pose=None, position=None):
        dim = corner.shape[0]
        self.corner = corner
        super().__init__(dim=dim, pose=pose, position=position)

    def _at_pose(self, pose, in_place):
        if not in_place:
            rect = AxisAlignedRect(self.high_corner, pose=pose)
        else:
            if np.any(pose.orientation != 0.0):
                raise NotImplementedError
            self.low_corner = pose.position
            self.high_corner = self.corner + pose.position

            rect = self
        return rect

    def _collides_with_rect(self, _):
        return True


class Point(MapObject):
    def _info_str(self):
        return ''

    def _at_pose(self, pose, in_place):
        if not in_place:
            point = Point(dim=self.dim, pose=pose)
        else:
            self.low_corner = pose.position
            self.high_corner = pose.position
            point = self
        return point

    def collides_with_rect(self, rect):
        if rect.dim != self.dim:
            raise ValueError

        position = self.pose.position
        return np.all(
            (rect.low_corner <= position) & (position <= rect.high_corner)
        )


class Sphere(MapObject):
    def __init__(self, r, dim=None, pose=None):
        self.radius = r
        super().__init__(dim=dim, pose=pose)

    def _info_str(self):
        return 'r=%.3f' % self.radius

    def _at_pose(self, pose, in_place):
        if not in_place:
            sphere = Sphere(r=self.radius, dim=self.dim, pose=pose)
        else:
            self.low_corner = pose.position - self.radius
            self.high_corner = pose.position + self.radius
            sphere = self
        return sphere

    def _collides_with_rect(self, rect):
        # check the largest set of coordinates of the sphere which
        # projects to the rect
        low_corner = rect.low_corner
        high_corner = rect.high_corner
        position = self.pose.position
        inds = (position < low_corner) | (high_corner < position)

        if not np.any(inds):
            # If all project to rect center is in rect
            return True

        corner = np.zeros(self.dim, dtype=float)
        lower = position < low_corner

        corner[lower] = low_corner[lower]
        corner[~lower] = high_corner[~lower]

        return np.sum(np.square(position[inds] - corner[inds])) < self.radius**2


class ConvexMesh(MapObject):
    pass


class Polygon(ConvexMesh):
    def __init__(self, *positions, keep_origin=False, pose=None):
        dim = positions[0].dim
        if dim != 2:
            raise ValueError

        mean = Position(np.zeros(dim, dtype=float))
        for position in positions:
            if position.dim != dim:
                raise ValueError
            mean.position -= position.position
        mean.position /= len(positions)

        if not keep_origin:
            self.points = [position.translate_by(mean) for position in positions]

        self.coords = np.zeros((len(self.points), dim), dtype=float)
        self.lines = np.zeros((len(self.points), dim + 1), dtype=float)
        super().__init__(dim, pose=pose)

    def _info_str(self):
        string = ''
        for i, p in enumerate(self.points):
            if i > 0:
                string += ', '
            string += 'p%d=%s' % (i, p)
        return string

    def _at_pose(self, pose, in_place):
        if not in_place:
            polygon = Polygon(*deepcopy(self.points), pose=pose)
        else:
            self.low_corner = np.full(self.dim, np.inf, dtype=float)
            self.high_corner = np.full(self.dim, -np.inf, dtype=float)
            for i, point in enumerate(self.points):
                point.align_to_pose(pose)
                self.coords[i] = point.position

                smaller = point.position < self.low_corner
                self.low_corner[smaller] = point.position[smaller]
                larger = self.high_corner < point.position
                self.high_corner[larger] = point.position[larger]

            for i, (p1, p2) in enumerate(
                zip(self.points, (self.points[-1], *self.points[:-1]))
            ):
                self.lines[i] = self._get_line(p1, p2)

            polygon = self
        return polygon

    def _get_line(self, point1, point2):
        rot = np.array([
            [0.0, -1.0],
            [1.0,  0.0]
        ])

        line = np.zeros(self.dim + 1, dtype=float)
        line[:-1] = rot @ (point2.position - point1.position)
        line[-1] = -np.dot(line[:-1], point1.position)

        return line

    def _collides_with_rect(self, rect):
        low_corner = rect.low_corner
        high_corner = rect.high_corner

        # Bounding box is inside rectangle
        bb_inside_rectangle = (
            np.all(low_corner < self.low_corner) and
            np.all(self.high_corner < high_corner)
        )
        if bb_inside_rectangle:
            return True

        # Any point is inside the rectangle
        for point in self.points:
            if np.all(
                (low_corner <= point.position) &
                (point.position <= high_corner)
            ):
                return True

        # Any corner of the rectangle is inside the polygon
        corners = np.array(
            np.meshgrid(low_corner, high_corner)
        ).T.reshape(-1, 1, self.dim)

        in_polygon = np.full(corners.shape[0], True)
        d, m = self.lines[:, :-1], self.lines[:, -1]

        d = d.reshape(1, -1, self.dim)
        m = m.reshape(1, -1)

        sign = np.sum(d * corners, axis=-1) + m
        in_polygon = np.all(sign <= 0.0, axis=-1) | np.all(sign >= 0.0, axis=-1)

        if np.any(in_polygon):
            return True

        # Any line of the polygon intersects with the box
        low_corner = low_corner.reshape(1, -1)
        high_corner = high_corner.reshape(1, -1)

        point1 = self.coords
        point2 = np.roll(self.coords, 1, axis=0)

        diff = point2 - point1
        temp1 = (low_corner - point1) / diff
        temp2 = (high_corner - point1) / diff

        tmin = np.minimum(temp1.min(axis=-1), temp2.min(axis=-1))
        tmax = np.maximum(temp1.max(axis=-1), temp2.max(axis=-1))

        intersects_box = (tmin <= 1.0) & (tmax >= 0.0) & (tmin <= tmax)

        return np.any(intersects_box)
