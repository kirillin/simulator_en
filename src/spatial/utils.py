import numpy as np


def skew(v):
    v = v.flatten()
    return np.array([
            [0,    -v[2],  v[1]],
            [v[2],  0,    -v[0]],
            [-v[1],  v[0],  0]
    ])


def unskew(A):
    return 0.5 * np.array([A[2,1] - A[1,2], A[0,2] - A[2,0], A[1,0] - A[0,1]])


def crm(v):
    return np.block([
        [skew(v[:3]), np.zeros((3,3))], 
        [skew(v[3:]), skew(v[:3])]
    ])

def crf(v):
    return -crm(v).T