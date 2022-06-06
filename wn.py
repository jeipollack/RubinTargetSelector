try:
    from numba import jit
except ImportError:
    jit = lambda func: func
import numpy as np

@jit
def is_left(p0, p1, p2):
    return ((p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p2[0] - p0[0]) * (p1[1] - p0[1]))


@jit
def wn_pnpoly(p, v):
    """Winding number of point p in polygon v.

    Note: v[0] == v[n-1] (closed polygon)
    """
    wn = 0
    n = len(v) - 1
    for i in range(n):
        if v[i, 1] <= p[1]:
            if v[i+1, 1] > p[1]:
                if is_left(v[i], v[i+1], p) > 0:
                    wn += 1
        elif v[i+1, 1] <= p[1]:
            if is_left(v[i], v[i+1], p) < 0:
                wn -= 1
    return wn


@jit
def wn_multipnpoly(ps, v):
    """Winding number of points ps in polygon v.

    Note: 
        v[0] == v[n-1] (closed polygon)
        ps.shape == (np , 2) with np the number of points

    """
    wn = np.zeros(len(ps), dtype=np.int32)
    for i, p in enumerate(ps):
        wn[i] = wn_pnpoly(p, v)
    return wn


def tests():
    import numpy as np
    v = np.array([[-1, -1],
                  [1, -1],
                  [0, 1],
                  [-1, -1]])
    p = np.array([0, 0])
    assert wn_pnpoly(p, v) == 1
    p = np.array([1, 1])
    assert wn_pnpoly(p, v) == 0


def demo():
    from matplotlib import pyplot as plt
    import numpy as np
    x = np.linspace(-2, 2, 21)
    y = x.copy()
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    v = np.array([[-1, -1],
                  [1, -1],
                  [0, 1],
                  [-1, -1]])
    res = wn_multipnpoly(xy, v)
    plt.scatter(xy[:, 0], xy[:, 1], c=res)
    plt.plot(v[:, 0], v[:, 1], '-o')
    plt.colorbar()
    plt.show()
    # input('Press ENTER to quit')

if __name__ == '__main__':
    demo()
