import numpy as np
from .occupancy import make_map


rng = np.random.default_rng()


def unpack_kwargs(dictionary, **kwargs):
    for key in dictionary:
        if key in kwargs:
            kwargs[key] = dictionary[key]
        else:
            raise ValueError('Unexpected keyword argument: %s' % key)
    return kwargs.values()


def sample_map(name=None, dim=2, **kwargs):
    if dim <= 0:
        raise ValueError('Dimension must be positive.')

    chart = make_map(
        np.array([0.0 for _ in range(dim)]),
        np.array([1.0 for _ in range(dim)])
    )
    if name == 'HalfSphere':
        n, smallest_mass = unpack_kwargs(
            kwargs,
            n=1000,
            smallest_mass=0.001
        )

        pts = rng.normal(0, 1, size=(dim, n))
        pts *= 0.3 / np.linalg.norm(pts, axis=0)
        pts[-1, pts[-1] < 0] *= -1
        pts += 0.5

        chart.fit_point_cloud(pts, smallest_mass=smallest_mass)
    else:
        raise ValueError('Sample map "%s" does not exist.' % name)

    return chart
