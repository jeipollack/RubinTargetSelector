import numpy as np


def to_vec(ang):
    """Convert angular coordinates to vector.

    ang: shape [N, 2], unit: degrees, format: lon, lat
    returns vec: shape [N, 3], unit: none, format: x, y, z (norm=1)
    """
    N = ang.shape[0]
    theta = np.pi/2 - np.radians(ang[:, 1])
    phi = np.radians(ang[:, 0])
    xyz = np.zeros((N, 3))
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    xyz[:, 0] = st*cp
    xyz[:, 1] = st*sp
    xyz[:, 2] = ct
    return xyz


def to_ang(vec):
    """Convert unit vectors to angular coordinates.
    
    vec: shape [N, 3], format: x, y, z
    returns ang: shape [N, 2], unit: degree, format: lon, lat
    """
    N = vec.shape[0]
    ang = np.zeros((N, 2))
    ang[:, 0] = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    r = np.sqrt((vec**2).sum(axis=1))
    c = vec[:, 2] / r
    ang[:, 1] = 90 - np.degrees(np.arccos(c))

    return ang


def rot_mat(lon, lat):
    c_lon, s_lon = np.cos(np.radians(lon)), np.sin(np.radians(lon))
    c_lat, s_lat = np.cos(np.radians(lat)), np.sin(np.radians(lat))
    r_y = np.array([
        [c_lat, 0, -s_lat],
        [0, 1, 0],
        [s_lat, 0, c_lat]
    ])
    r_z = np.array([
        [c_lon, -s_lon, 0],
        [s_lon, c_lon, 0],
        [0, 0, 1]
    ])
    return np.matmul(r_z,r_y)


def rotate_to(lon, lat, ang):
    r = rot_mat(lon, lat)
    vec = np.matmul(to_vec(ang),r.T)
    return to_ang(vec)
