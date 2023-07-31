import numpy as np
from occupancy_map.abstract import CellState
from occupancy_map.pose import Pose


rng = np.random.default_rng()


class PlannerRRT:
    def __init__(
        self,
        chart,
        obj,
        growth_rate=0.9,
        step_size=1e-1,
        max_iters=1000
    ):
        self.map = chart
        self.dim = self.map.dim
        self.object = obj

        self.growth_rate = growth_rate
        self.step_size = step_size
        self.max_iters = max_iters

        self.iters = 0
        self.reached = False

    def _sample_position_in_map(self):
        return Pose(
            rng.uniform(self.map.low_corner, self.map.high_corner)
        )

    def _sample_object_in_map(self):
        collides = True
        while collides:
            position = self._sample_position_in_map()
            sampled = self.object.at_pose(position, in_place=False)

            collides = self.map.object_state(
                sampled,
                [CellState.unknown, CellState.occupied]
            )
        return sampled

    def plan(self, from_point, to_point):
        self.reached = False
        self.iters = 0

        from_object = self.object.at_pose(pose=from_point.pose, in_place=False)
        to_object = self.object.at_pose(pose=to_point.pose, in_place=False)

        if (
            self.map.object_state(
                from_object, [CellState.unknown, CellState.occupied]
            ) or
            self.map.object_state(
                to_object, [CellState.unknown, CellState.occupied]
            )
        ):
            return []

        self.tree = {from_object: None}
        return self._plan(self.tree, to_object)

    def _plan(self, tree, goal):
        reached = False

        while not reached and self.iters < self.max_iters:
            if rng.uniform() < self.growth_rate:
                sample = self._sample_object_in_map()
                self._grow_tree(tree, sample, grow=True)
            else:
                reached = self._grow_tree(tree, goal, grow=False)

        if goal not in tree:
            return []

        curr = goal
        prev = tree[goal]
        path = [goal]
        while prev is not None:
            path.append(prev)
            curr = prev
            prev = tree[curr]

        return list(reversed(path))

    def _grow_tree(self, tree, sample, grow=True):
        self.iters += 1
        closest = self._closest_object(tree, sample)

        reached = False
        while not reached and closest is not None:
            closest, reached = self._extend(tree, closest, sample)
            temp = self._closest_object(tree, sample)
            if closest is not temp:
                break

        return reached

    def _extend(self, tree, closest, sample):
        p1 = closest.pose.position
        p2 = sample.pose.position
        direction = p2 - p1
        magnitude = np.linalg.norm(direction)

        if magnitude < self.step_size:
            reached = True
            node = sample
        else:
            reached = False
            p = p1 + self.step_size * direction / magnitude
            node = self.object.at_pose(Pose(p), in_place=False)

        unobstructed = self._check_path(closest, node)

        if unobstructed:
            tree[node] = closest
            closest = node
        else:
            closest = None

        return closest, reached

    def _check_path(self, from_node, to_node):
        n = 10
        p1 = from_node.pose.position
        p2 = to_node.pose.position
        ps = np.linspace(p1, p2, n)[1:-1]

        for i in range(n-2):
            p = ps[i]
            middle_node = self.object.at_pose(pose=Pose(p), in_place=False)
            if self.map.object_state(
                middle_node, [CellState.unknown, CellState.occupied]
            ):
                return False

        return True

    def _closest_object(self, tree, sample):
        closest = None
        dist = np.inf
        for obj in tree:
            temp = np.square(
                obj.pose.position - sample.pose.position
            ).sum()
            if temp < dist:
                closest = obj
                dist = temp
        return closest
