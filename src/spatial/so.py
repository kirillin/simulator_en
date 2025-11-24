import numpy as np
from spatial.utils import skew, unskew


class so3:
    def __init__(self, w):
        self.w = np.array(w, dtype=float).flatten()

    def hat(self):
        # R^3 -> so3
        w = np.array(self.w).reshape(3, 1)
        return skew(w)

    def vee(hat_w):
        # so3 -> R^3
        return unskew(hat_w)

    def exp(self):
        # so3 -> SO3
        theta = np.linalg.norm(self.w)
        if theta < 1e-9:
            return SO3()  # near zero → identity
        axis = self.w / theta
        K = so3.hat(axis)
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
        return SO3(R)


class SO3:
    def __init__(self, R=None):
        self._R = np.eye(3) if R is None else np.array(R, dtype=float)

    @property
    def matrix(self):
        return self._R

    @staticmethod
    def identity():
        return SO3()
    
    def compose(self, other):
        return SO3(self_R @ other.matrix)
    
    def inverse(self):
        return SO3(self._R.T)
    
    def act(self, p):
        return self._R @ np.array(p).flatten()
    
    @staticmethod
    def from_axis_angle(axis, angle):
        axis = np.array(axis).flatten()
        axis = axis / np.linalg.norm(axis)
        return se3(axis).exp()

    @staticmethod
    def rx(angle):
        return SO3.from_axis_angle((1,0,0), angle).matrix

    @staticmethod
    def ry(angle):
        return SO3.from_axis_angle((0,1,0), angle).matrix

    @staticmethod
    def rz(angle):
        return SO3.from_axis_angle((0,0,1), angle).matrix
