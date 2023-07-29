import sys
sys.path.append('..')

from occupancy_map.pose import Pose
from occupancy_map.map_object import Point, Sphere
from occupancy_map.sample import sample_map
from planner.rrt import PlannerRRT
from planner.repeated_rrt import RepeatedPlanner
from visualize.visualize import MapWindow


def draw_tree(window, tree):
    for curr, prev in tree.items():
        if prev is None:
            continue
        p1 = prev.pose.position
        p2 = curr.pose.position
        window.ax.plot(
            (p1[1], p2[1]),
            (p1[0], p2[0]),
            '-*',
            color='black'
        )


def main2d():
    window = MapWindow(dim=2)
    chart = sample_map(name='HalfSphere', dim=2)

    obj = Sphere(r=0.05, dim=2)
    planner = PlannerRRT(chart, obj)
    rep = RepeatedPlanner(planner)

    p1 = Point(pose=Pose([0.5, 0.7]))
    p2 = Point(pose=Pose([0.5, 0.9]))

    # path = planner.plan(p1, p2)
    path = rep.plan(p1, p2)
    # print(tree)

    window.plot_map(chart)
    # draw_tree(window, planner.tree)
    window.plot_objs(p1, p2, *path)
    window.show()


def main3d():
    window = MapWindow(dim=3)
    chart = sample_map(name='HalfSphere', dim=3)

    obj = Sphere(r=0.05, dim=3)
    planner = PlannerRRT(chart, obj)
    rep = RepeatedPlanner(planner, total_plans=100)

    p1 = Point(pose=Pose([0.5, 0.5, 0.7]))
    p2 = Point(pose=Pose([0.5, 0.5, 0.9]))

    # path = planner.plan(p1, p2)
    path = rep.plan(p1, p2)
    # print(tree)

    window.plot_map(chart)
    # draw_tree(window, planner.tree)
    window.plot_objs(p1, p2, *path)
    window.show()


if __name__ == '__main__':
    # main2d()
    main3d()
