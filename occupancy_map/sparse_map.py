from occupancy import CellState


def prod(nums):
    prod = 1
    for num in nums:
        prod *= num
    return prod


class Cell:
    def __init__(self, state):
        self.state = state


class SparseMap(Cell):
    def __init__(self, background, dims):
        self.background = background
        self.states = {background}

        self._cells = dict()

    def add_cell(self, cell, ind):
        self._cells[ind] = cell
        self.contents.add(cell.state)
