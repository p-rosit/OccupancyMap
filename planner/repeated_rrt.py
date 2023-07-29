import numpy as np


class RepeatedPlanner:
    def __init__(
        self,
        planner,
        total_plans=20
    ):
        self.planner = planner
        self.total_plans = total_plans

    def plan(self, from_point, to_point):
        shortest_plan = []
        shortest = np.inf

        for _ in range(self.total_plans):
            plan = self.planner.plan(from_point, to_point)
            length = self._path_length(plan)

            if length < shortest:
                shortest = length
                shortest_plan = plan

        return shortest_plan

    def _path_length(self, path):
        if not path:
            return np.inf
        return sum(
            np.linalg.norm(
                obj1.pose.position - obj2.pose.position
            )
            for obj1, obj2 in zip(path, [path[-1], *path[:-1]])
        )
