import numpy as np

class PhysicsEngine:
    def __init__(self, fd_solver, gravity=[0.0, -9.81, 0.0]):
        self.fd_solver = fd_solver
        self.gravity = np.array(gravity)

    def update(self, objects, plane, dt):
        for obj in objects:

            qdd = self.fd_solver.forward_dynamics(obj, q, qd, tau)

            robot.qd += qdd * dt
            robot.q += robot.qd * dt