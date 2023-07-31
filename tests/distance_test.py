import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from occupancy_map.pose import Pose
from occupancy_map.map_object import Point
from occupancy_map.abstract import CellState
from occupancy_map.sample import sample_map
from visualize.visualize import MapWindow


def main2d():
    window = MapWindow(dim=2)
    chart = sample_map(name='HalfSphere', dim=2, n=10000, smallest_mass=0.0001)

    n = 100
    x = np.linspace(chart.low_corner[0], chart.high_corner[0], n)
    y = np.linspace(chart.low_corner[1], chart.high_corner[1], n)
    z = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            z[i, j] = chart.smallest_distance(
                Point(pose=Pose([x[i], y[j]], [0.0])),
                CellState.occupied
            )

    window.plot_map(chart)
    window.draw()

    fig = plt.figure()
    plt.imshow(z)
    plt.show()


def main3d():
    # window = MapWindow(dim=3)
    chart = sample_map(name='HalfSphere', dim=3, n=100000000, smallest_mass=0.00001)

    n = 200
    m = 4
    x = np.linspace(chart.low_corner[0], chart.high_corner[0], m)
    y = np.linspace(chart.low_corner[1], chart.high_corner[1], n)
    z = np.linspace(chart.low_corner[2], chart.high_corner[2], n)
    d = np.zeros((m, n, n), dtype=float)

    for i in range(m):
        print(i)
        for j in range(n):
            for k in range(n):
                d[i, j, k] = chart.smallest_distance(
                    Point(pose=Pose([x[i], y[j], z[k]], [0.0, 0.0, 0.0])),
                    CellState.occupied
                )

    # window.plot_map(chart)
    # window.draw()

    ind = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    slider = Slider(
        plt.axes([0.25, 0.1, 0.65, 0.03]),
        label='height',
        valmin=chart.low_corner[ind],
        valmax=chart.high_corner[ind],
        valinit=0.0,
        closedmin=True,
        closedmax=False,
        valstep=(chart.high_corner[ind] - chart.low_corner[ind]) / m
    )

    def update(val):
        index = val - chart.low_corner[ind]
        index = int(m * index / (chart.high_corner[ind] - chart.low_corner[ind]))

        ax.imshow(d[index])

    slider.on_changed(update)

    ax.imshow(d[0])
    ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    main2d()
    main3d()
