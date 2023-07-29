import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as patches

from occupancy_map.occupancy import CellState
from occupancy_map.map_object import AxisAlignedRect, Point, Sphere, Polygon


class MapWindow:
    def __init__(self, dim=2):
        self.fig = plt.figure()
        self.dim = dim

        if dim == 2:
            self.ax = self.fig.add_subplot()
        elif dim == 3:
            self.ax = self.fig.add_subplot(projection='3d')
        else:
            raise ValueError('Cannot plot map of dimension other than 2 or 3.')

    def plot_map(self, chart):
        if chart.dim != self.dim:
            raise ValueError('Map window is %d-dimensional while input map is %d-dimensional.' % (self.dim, chart.dim))

        if chart.dim == 2:
            self.ax.axis('equal')
            self.ax.set_xlim((chart.low_corner[1], chart.high_corner[1]))
            self.ax.set_ylim((chart.low_corner[0], chart.high_corner[0]))

            _plot_2d_map(chart, self.ax)
        else:
            self.ax.set_xlim(chart.low_corner[0], chart.high_corner[0])
            self.ax.set_ylim(chart.low_corner[1], chart.high_corner[1])
            self.ax.set_zlim(chart.low_corner[2], chart.high_corner[2])

            _plot_3d_map(chart, self.ax)

    def plot_obj(self, obj):
        if obj.abstract:
            raise ValueError('Cannot plot an abstract object.')
        if obj.dim != self.dim:
            raise ValueError('Map window is %d-dimensional while input object is %d-dimnesional.' % (self.dim, obj.dim))

        if isinstance(obj, AxisAlignedRect):
            plot_axis_aligned_rect(obj, self.ax)
        if isinstance(obj, Point):
            plot_point(obj, self.ax)
        elif isinstance(obj, Sphere):
            plot_sphere(obj, self.ax)
        elif isinstance(obj, Polygon):
            plot_polygon(obj, self.ax)
        else:
            raise ValueError('Cannot handle map object of type "%s".' % obj.__class__.__name__)

    def plot_objs(self, *objs):
        for obj in objs:
            self.plot_obj(obj)

    def draw(self):
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show():
        plt.show()


def plot_axis_aligned_rect(rect, ax, col=(0.0, 0.0, 0.0)):
    if rect.dim == 2:
        lox, loy = rect.low_corner
        hix, hiy = rect.high_corner

        rect = patches.Rectangle(
            (loy, lox),
            hiy - loy,
            hix - lox,
            edgecolor='black',
            facecolor=col
        )
        ax.add_patch(rect)
    elif rect.dim == 3:
        _plot_cube(rect, ax, col=col)
    else:
        raise ValueError


def plot_point(point, ax):
    position = point.pose.position
    if point.dim == 2:
        ax.plot(position[1], position[0], '*')
    elif point.dim == 3:
        ax.plot(position[2], position[0], position[1], '*')
    else:
        ValueError


def plot_sphere(sphere, ax):
    point, radius = sphere.pose.position, sphere.radius

    if sphere.dim == 2:
        circ = plt.Circle(
            (point[1], point[0]),
            radius,
            color='black',
            fill=False
        )
        ax.add_patch(circ)
    else:
        # Make data
        u = np.linspace(0, 2 * np.pi, 9)
        v = np.linspace(0, np.pi, 9)
        x = point[2] + radius * np.outer(np.cos(u), np.sin(v))
        y = point[0] + radius * np.outer(np.sin(u), np.sin(v))
        z = point[1] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface
        ax.plot_surface(x, y, z, color='black', alpha=0.5)


def plot_line(point1, point2, ax):
    if len(point1) not in [2, 3]:
        raise ValueError
    if len(point2) not in [2, 3]:
        raise ValueError
    if len(point1) != len(point2):
        raise ValueError

    if len(point1) == 2:
        x1, y1 = point1
        x2, y2 = point2
        plt.plot((y1, y2), (x1, x2), color='black')
    else:
        raise NotImplementedError


def plot_polygon(polygon, ax):
    points = np.zeros((polygon.dim, len(polygon.points)), dtype=float)
    for i, point in enumerate(polygon.points):
        points[:, i] = point.position

    if points.shape[0] == 2:
        ax.fill(points[1], points[0], color='black', fill=False)
    else:
        raise ValueError


def _plot_2d_map(cell, ax):
    ax.set_xlabel('x_2')
    ax.set_ylabel('x_1')

    if cell.state is CellState._recurse:
        for row in cell.grid:
            for elm in row:
                _plot_2d_map(elm, ax)
    else:
        lox, loy = cell.low_corner
        hix, hiy = cell.high_corner

        col = CellState.col(cell.state)
        plot_axis_aligned_rect(cell.bounding_box, ax, col=col)


def _plot_3d_map(cell, ax=None):
    ax.set_xlabel('x_3')
    ax.set_ylabel('x_1')
    ax.set_zlabel('x_2')

    if cell.state is CellState._recurse:
        for plane in cell.grid:
            for row in plane:
                for elm in row:
                    _plot_3d_map(elm, ax)
    else:
        col = CellState.col(cell.state, transparent=True)
        if col[-1] > 0:
            plot_axis_aligned_rect(cell.bounding_box, ax, col=col)


def _plot_cube(rect, ax, col=(0.0, 0.0, 0.0)):
    lox, loy, loz = rect.low_corner
    hix, hiy, hiz = rect.high_corner

    for axis in ['x', 'y', 'z']:
        if axis == 'y':
            lox, loy = loy, lox
            hix, hiy = hiy, hix

        for val in [loz, hiz]:
            rect = patches.Rectangle(
                (lox, loy),
                hix - lox,
                hiy - loy,
                facecolor=col
            )

            ax.add_patch(rect)
            art3d.pathpatch_2d_to_3d(rect, z=val, zdir=axis)

        if axis == 'y':
            lox, loy = loy, lox
            hix, hiy = hiy, hix

        lox, loy, loz = (loy, loz, lox)
        hix, hiy, hiz = (hiy, hiz, hix)
