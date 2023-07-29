import sys
sys.path.append('..')

from occupancy_map.occupancy import CellState
from occupancy_map.sample import sample_map
from occupancy_map.pose import Pose
from occupancy_map.map_object import Sphere, Point
from visualize.visualize import MapWindow


def main():
    chart2 = sample_map('HalfSphere', dim=2, n=1000, smallest_mass=0.001)
    chart3 = sample_map('HalfSphere', dim=3, n=10000, smallest_mass=0.001)

    inds2 = [0, 1, 2, 3, 4, 5, 6, 7]
    objs2 = [
        Point(pose=Pose([0.5, 0.5], [0.0])),              # 0
        Point(pose=Pose([0.3, 0.6], [0.0])),              # 1
        Point(pose=Pose([0.7, 0.4], [0.0])),              # 2
        Point(pose=Pose([0.2, 0.52], [0.0])),             # 3
        Sphere(r=0.155, pose=Pose([0.7, 0.3], [0.0])),      # 4
        Sphere(r=0.05, pose=Pose([0.258, 0.465], [0.0])),   # 5
        Sphere(r=0.03, pose=Pose([0.76, 0.5], [0.0])),      # 6
        Sphere(r=0.4, pose=Pose([0.5, 0.5], [0.0])),        # 7
    ]

    inds3 = [0, 1, 2]
    objs3 = [
        Point(pose=Pose([0.5, 0.5, 0.5], [0.0, 0.0, 0.0])),         # 0
        Sphere(r=0.2, pose=Pose([0.5, 0.5, 0.5], [0.0, 0.0, 0.0])),   # 1
        Sphere(r=0.05, pose=Pose([0.3, 0.3, 0.6], [0.0, 0.0, 0.0]))   # 2
    ]

    for inds, objs, window, chart in zip(
        [inds2, inds3],
        [objs2, objs3],
        [MapWindow(dim=2), MapWindow(dim=3)],
        [chart2, chart3]
    ):
        print('Dimension:', chart.dim)
        for ind in inds:
            print(
                '%s:' % objs[ind],
                chart.object_state(objs[ind], CellState.occupied)
            )

        window.plot_map(chart)
        window.plot_objs(*[objs[ind] for ind in inds])
        window.draw()

    MapWindow.show()


if __name__ == '__main__':
    main()
