import sys
sys.path.append('..')

import numpy as np

from occupancy_map.pose import Pose, Position
from occupancy_map.occupancy import CellState
from occupancy_map.sample import sample_map
from occupancy_map.map_object import Sphere, Polygon
from visualize.visualize import MapWindow


rng = np.random.default_rng()


def sample_pose(chart):
    pos = rng.uniform(chart.low_corner, chart.high_corner)

    ornt = rng.random(size=((chart.dim - 1) * chart.dim) // 2)
    sample_mag = rng.uniform(0.0, np.pi)
    ornt *= sample_mag / np.linalg.norm(ornt)

    return Pose(pos, ornt)


def sample_object(chart, obj):
    collides = True

    while collides:
        pose = sample_pose(chart)
        sampled = obj.at_pose(pose, in_place=False)

        collides = chart.object_state(
            sampled,
            [CellState.unknown, CellState.occupied]
        )

    return sampled


def main2d():
    dim = 2
    window = MapWindow(dim=dim)
    chart = sample_map(name='HalfSphere', dim=dim, smallest_mass=0.001)

    sphere = Sphere(r=0.1, dim=dim)
    sampled = []
    for _ in range(5):
        sampled.append(sample_object(chart, sphere))

    polygon = Polygon(
        Position([0.0, 0.0]),
        Position([0.1, 0.0]),
        Position([0.0, 0.1])
    )
    polygon = Polygon(
        Position([0.0, 0.0]),
        Position([0.0, 0.35]),
        Position([0.25, 0.35]),
        Position([0.3, 0.15]),
        Position([0.25, 0.0])
    )

    sampled = []
    for _ in range(500):
        sampled.append(sample_object(chart, polygon))

    window.plot_map(chart)
    window.plot_objs(*sampled)
    window.show()


if __name__ == '__main__':
    main2d()
