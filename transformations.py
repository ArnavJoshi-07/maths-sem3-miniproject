"""A linear transformation acts on each column point of a 2xN shape."""
import numpy as np
import matrix_ops as ops


def translate(tx, ty):
    # Translation is affine, so it has no 2x2 linear transformation matrix.
    # This column is repeated once per corner before matrix addition.
    return np.array([[tx], [ty]], dtype=float)


def translation_offset(tx, ty, shape):
    return np.repeat(translate(tx, ty), shape.shape[1], axis=1)


def scale(k):
    return ops.scalar_multiply(np.eye(2), k)


def scale_xy(sx, sy):
    return np.diag([float(sx), float(sy)])


def rotate(theta_degrees):
    theta = np.deg2rad(theta_degrees)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def reflect(axis):
    choices = {'x': [[1,0],[0,-1]], 'y': [[-1,0],[0,1]],
               'y=x': [[0,1],[1,0]]}
    if axis not in choices:
        raise ValueError('Reflection axis must be x, y, or y=x.')
    return np.array(choices[axis], dtype=float)


def shear(shx, shy):
    return np.array([[1, shx], [shy, 1]], dtype=float)


def project_onto_x():
    return np.array([[1., 0.], [0., 0.]])


def apply(T, shape):
    T, shape = ops.matrix(T), ops.matrix(shape)
    if T.shape != (2,2) or shape.shape[0] != 2:
        raise ValueError('Use a 2x2 transformation and a 2xN shape.')
    print('Shape matrix:\n', shape, '\nTransformation T:\n', T)
    return ops.multiply(T, shape)


def undo(T, shape):
    inverse = ops.inverse_gauss_jordan(T)
    if inverse is None:
        print('Information was lost: crushed coordinates cannot be recovered uniquely.')
        return None
    return apply(inverse, shape)


def rotation_orthogonal(R):
    print('Rotation transpose:\n', ops.transpose(R))
    inverse = ops.inverse_gauss_jordan(R)
    if inverse is not None:
        print('For rotation matrices, transpose(R) = inverse(R).')
        ops.verified(R.T, inverse)
