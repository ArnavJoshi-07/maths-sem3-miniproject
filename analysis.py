"""Explain area, systems of equations, subspaces, eigenvectors and SVD."""
import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve
from sympy import Matrix
import matplotlib.pyplot as plt
import matrix_ops as ops
import visualize as vis


def area(shape):
    # Shoelace formula: vertices must follow the polygon boundary in order.
    x, y = shape
    return abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))) / 2


def det_meaning(T, shape=None):
    d = ops.determinant(T)
    print('Area scale factor = |det(T)| =', abs(d))
    print('Mirrored orientation.' if d < 0 else 'Orientation preserved.' if d > 0 else 'Flattened: area becomes zero.')
    if shape is None:
        shape = np.array([[0.,2.,2.,0.],[0.,0.,2.,2.]])
    before, after = area(shape), area(T @ shape)
    print('Shoelace area before:', before, '\nShoelace area after:', after)
    ops.verified(after, abs(d)*before)
    return d


def find_transformation(p1, p2, q1, q2):
    # Unknown vector [a,b,c,d] describes T = [[a,b],[c,d]].
    p1, p2, q1, q2 = [np.asarray(p, dtype=float).reshape(2) for p in (p1,p2,q1,q2)]
    x1,y1 = p1; x2,y2 = p2
    A = np.array([[x1,y1,0,0],[0,0,x1,y1],[x2,y2,0,0],[0,0,x2,y2]])
    b = np.concatenate((q1,q2))
    print('Unknowns: [a,b,c,d]; T = [[a,b],[c,d]]')
    print('A:\n', A, '\nb:\n', b)
    r, augmented_rank = np.linalg.matrix_rank(A), np.linalg.matrix_rank(np.column_stack((A,b)))
    print('rank(A) =', r, '; rank([A|b]) =', augmented_rank)
    if r < 4:
        print('no solution' if r < augmented_rank else 'infinitely many solutions')
        # Least squares gives a representative fit and its residual, not a unique solution.
        x = np.linalg.lstsq(A,b,rcond=None)[0]
        residual = A @ x - b
        print('Least-squares representative:', x, '\nResidual error:', np.linalg.norm(residual))
        ops.verified(np.allclose(A @ x, b), r == augmented_rank)
        return None
    x = solve(A,b)
    print('Direct solution:', x)
    lu, piv = lu_factor(A)
    print('Packed LU (U above diagonal, L below with implicit unit diagonal):\n', lu)
    print('LU pivot swaps:', piv)
    x_lu = lu_solve((lu,piv),b)
    print('LU solution:', x_lu)
    ops.verified(x, x_lu)
    print('Residual A@x-b:', A @ x-b, '\nResidual error:', np.linalg.norm(A @ x-b))
    ops.verified(A @ x,b)
    T = x.reshape(2,2)
    print('Recovered T:\n', T)
    ops.verified(T @ np.column_stack((p1,p2)), np.column_stack((q1,q2)))
    return T


def subspaces(T):
    S = Matrix(T)
    spaces = [('Column space', S.columnspace()), ('Row space', S.rowspace()),
              ('Null space', S.nullspace()), ('Left null space', S.T.nullspace())]
    for label, basis in spaces:
        print(label, 'basis:', basis if basis else '{0} (empty basis)')
    # Null vectors are killed by T; left-null vectors are killed by T transpose.
    for v in S.nullspace():
        ops.verified(T @ np.array(v,dtype=float), np.zeros((2,1)))
    for v in S.T.nullspace():
        ops.verified(T.T @ np.array(v,dtype=float), np.zeros((2,1)))
    ops.verified(len(S.columnspace()) + len(S.nullspace()), 2)
    ops.verified(len(S.rowspace()) + len(S.T.nullspace()), 2)
    if np.allclose(T, [[1,0],[0,0]]):
        print('Projection: null space is the y direction that gets crushed;')
        print('column space is the x-axis line the shape lands on.')
    return spaces


def eigen(T, filename=None):
    values, vectors = np.linalg.eig(T)
    print('Eigenvalues:', values, '\nEigenvectors (columns):\n', vectors)
    for i, value in enumerate(values):
        ops.verified(T @ vectors[:,i], value*vectors[:,i])
    if np.any(np.abs(values.imag) > 1e-9):
        print('Complex eigenvalues: no real eigenvector directions to draw for this rotation.')
        return
    fig, ax = plt.subplots(figsize=(7,6))
    for i, value in enumerate(values.real):
        v = vectors[:,i].real
        w = T @ v
        print('Eigenpair:', value, v, '; transformed:', w)
        ax.quiver(0,0,*v,angles='xy',scale_units='xy',scale=1,color='blue',label=f'v{i+1}')
        ax.quiver(0,0,*w,angles='xy',scale_units='xy',scale=1,color='red',alpha=.6,label=f'Tv{i+1}')
    limit = max(1., float(np.max(np.abs(T @ vectors.real)))) * 1.5
    ax.set_xlim(-limit,limit); ax.set_ylim(-limit,limit)
    ax.axhline(0,color='gray'); ax.axvline(0,color='gray'); ax.grid(True)
    ax.set_aspect('equal'); ax.legend()
    ax.set_title('Eigenvectors stay on the same line\nNegative eigenvalue reverses direction; zero crushes the vector')
    vis.finish(fig, filename)


def svd(T, shape=None, filename=None):
    if shape is None:
        shape = np.array([[0.,2.,2.,0.],[0.,0.,2.,2.]])
    U, s, VT = np.linalg.svd(T)
    Sigma = np.diag(s)
    print('U:\n', U, '\nSigma:\n', Sigma, '\nVT:\n', VT)
    print('Reconstruction U @ Sigma @ VT:\n', U @ Sigma @ VT)
    ops.verified(U @ Sigma @ VT, T)
    print('U and VT are orthogonal: they may rotate AND/OR reflect.')
    stages = [shape, VT @ shape, Sigma @ VT @ shape, U @ Sigma @ VT @ shape]
    for label, points in zip(['Original','After VT','After Sigma','After U'],stages):
        print(label, '\n', points)
    ops.verified(stages[-1],T @ shape)
    fig, axes = plt.subplots(1,4,figsize=(16,5))
    for ax, points, label in zip(axes,stages,['Original','VT: rotate/reflect','Sigma: stretch','U: rotate/reflect = final']):
        vis.draw(ax,shape,points,label)
    vis.finish(fig, filename)
