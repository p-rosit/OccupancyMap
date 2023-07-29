import sys
sys.path.append('..')

from occupancy_map.sample import sample_map
from occupancy_map.map_object import Sphere
from occupancy_map.pose import Pose
from visualize.visualize import MapWindow


def main():
    dim = 2

    chart = sample_map(name='HalfSphere', dim=dim, n=10000)

    sphere = Sphere(
        r=0.2,
        pose=Pose(
            [0.5 for _ in range(dim)],
            [0.0 for _ in range(((dim - 1) * dim) // 2)]
        )
    )

    window = MapWindow(dim=dim)
    window.plot_map(chart)
    window.plot_obj(sphere)
    window.show()


if __name__ == '__main__':
    main()
