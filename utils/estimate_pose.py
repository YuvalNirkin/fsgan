""" 3D pose estimation utilities. """

import numpy as np
from math import cos, sin, atan2, asin


# Adapted from: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
def isRotationMatrix(R):
    """ Checks if a matrix is a valid rotation matrix (whether orthogonal or not)

    Args:
        R (np.array): A matrix of shape (3, 3)

    Returns:
        bool: Whether the matrix R is a rotation matrix or not.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Adapted from: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
def matrix2angle(R):
    """ Compute three Euler angles from a Rotation Matrix.

    Ref: http://www.gregslabaugh.net/publications/euler.pdf

    Args:
        R (np.array): Rotation matrix of shape (3, 3)

    Returns:
        (float, float, float): A tuple containing the computed euler angles:
            - yaw (float)
            - pitch (float)
            - roll (float)
    """
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # Can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


# Adapted from: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
def P2sRt(P):
    """ Decompositing camera matrix P.

    Args:
        P (np.array): Affine Camera Matrix of shape (3, 4)

    Returns:
        (float, np.array, np.array): Tuple containing:
            - s (float): Scale factor
            - R (np.array): Rotation matrix of shape (3, 3)
            - t2d (np.array): 2D translation vector
    """
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d


# Adapted from: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
def compute_similarity_transform(points_static, points_to_transform):
    """ Compute similarity transform between two sets of points.

    Ref: http://nghiaho.com/?page_id=671

    Args:
        points_static (np.array): Target points set of shape (N, 3)
        points_to_transform (np.array): Source points set of shape (N, 3)

    Returns:
        - P (np.array): Projection matrix of shape (3, 4)
    """
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3, 1)
    t1 = -np.mean(p1, axis=1).reshape(3, 1)
    t_final = t1 - t0

    p0c = p0 + t0
    p1c = p1 + t1

    covariance_matrix = p0c.dot(p1c.T)
    U, S, V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0) ** 2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0) ** 2))

    s = (rms_d0 / rms_d1)
    P = np.c_[s * np.eye(3).dot(R), t_final]

    return P


# Adapted from: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
def estimate_pose(vertices):
    """ Estimate the pose of a set of points compared to a static canonical set of points.

    Args:
        vertices (np.array): A set of points to compute its pose

    Returns:
        (np.array, tuple of float): Tuple containing:
            - P (np.array): Camera matrix of shape (3, 4)
            - pose (tuple of float): The vertices pose euler angles: yaw, pitch, and roll
    """
    canonical_vertices = np.load('Data/uv-data/canonical_vertices.npy')
    P = compute_similarity_transform(vertices, canonical_vertices)
    _, R, _ = P2sRt(P)  # decompose affine matrix to s, R, t
    pose = matrix2angle(R)

    return P, pose


def euler2mat(angles):
    """ Convert euler angles to rotation matrix.

    Args:
        angles (tuple of float): Euler angles: yaw, pitch, and roll [Radians]

    Returns:
        R (np.array): Rotation matrix of shape (3, 3)
    """
    X = np.eye(3)
    Y = np.eye(3)
    Z = np.eye(3)

    x = angles[2]
    y = angles[1]
    z = angles[0]

    X[1, 1] = cos(x)
    X[1, 2] = -sin(x)
    X[2, 1] = sin(x)
    X[2, 2] = cos(x)

    Y[0, 0] = cos(y)
    Y[0, 2] = sin(y)
    Y[2, 0] = -sin(y)
    Y[2, 2] = cos(y)

    Z[0, 0] = cos(z)
    Z[0, 1] = -sin(z)
    Z[1, 0] = sin(z)
    Z[1, 1] = cos(z)

    R = Z @ Y @ X

    return R


def z1y2x3(alpha, beta, gamma):
    """ Compute Z1Y2X3 rotation matrix given Euler angles.

    Ref: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    Args:
        alpha (float): Rotation angle around the X axis [Radians]
        beta (float): Rotation angle around the Y axis [Radians]
        gamma (float): Rotation angle around the Z axis [Radians]

    Returns:
         R (np.array): The computed rotation matrix of shape (3, 3)
    """
    return np.array([[np.cos(alpha) * np.cos(beta),
                      np.cos(alpha) * np.sin(beta) * np.sin(gamma) -
                      np.cos(gamma) * np.sin(alpha),
                      np.sin(alpha) * np.sin(gamma) +
                      np.cos(alpha) * np.cos(gamma) * np.sin(beta)],
                     [np.cos(beta) * np.sin(alpha),
                      np.cos(alpha) * np.cos(gamma) +
                      np.sin(alpha) * np.sin(beta) * np.sin(gamma),
                      np.cos(gamma) * np.sin(alpha) * np.sin(beta) -
                      np.cos(alpha) * np.sin(gamma)],
                     [-np.sin(beta), np.cos(beta) * np.sin(gamma),
                      np.cos(beta) * np.cos(gamma)]])


# Adapted from: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
def rigid_transform_3D(A, B):
    """ Compute a rigid transform between two sets of 3D points.

    Ref: http://nghiaho.com/?page_id=671

    Args:
        A (np.array): Source set of 3D points of shape (3, N)
        B (np.array): Target set of 3D points of shape (3, N)

    Returns:
        (np.array, np.array): A tuple containing:
            - R (np.array): Rotation matrix of shape (3, 3)
            - t (np.array): 3D translation vector
    """
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) @ BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print
        "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    return R, t
