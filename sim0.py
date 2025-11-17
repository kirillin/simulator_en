import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 1. SRP
# #   - world
#         for i in range(1000)
#             world.add_object(disk)

# #   - rendering
#         render(r)

# #   - phisics engine
#         dv = some_algo(forces, torque)
#         v = ..
#         r = ..

# #   - data/state class
#         disk = Disk(R, m , I)

# 2. OCP
#   - Shape:
#       def area(): pass
#       def draw(): pass
#
#       - Triangle
#            def area():
#
#       - Square 
#           def area():

# 3. LSP
#   Shape obj1 = Triangle()
#   Shape obj2 = Square()
#   arr = [obj1, obj2]
#   for o in arr:
#       o.draw()

# 4. ISP
#   Vihecle
#       def drive()
#       def fly()
#
#           Car
#               def drive()
#               def fly() empty
#
#           Plane
#               def drive() empty
#               def fly()

# CarIf()
#     def drive()
#           Car
#               def drive()

# PlaneIf()
#   def fly()
#           Plane
#               def fly()

# 5. DIP

# highlevev control
# api
# lowlevel controller

# KISS
# YAGNI
# DRY


R = 0.2
m = 1.0
I = 0.4 * m * R**2

R2 = 0.3
m2 = 1.0
I2 = 0.4 * m2 * R2**2

R3 = 0.5
m3 = 1.0
I3 = 0.4 * m2 * R2**2

g = 9.81

dt = 0.01

# r == y in bb
r = np.array([0.0, 4.0])
v = np.array([0.0, 0.5])
omega = 5.0

r2 = np.array([2.0, 4.0])
v2 = np.array([0.0, 0.0])
omega2 = 5.0

r3 = np.array([4.0, 4.0])
v3 = np.array([0.0, 1.0])
omega3 = 5.0

fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')

ball_patch = Circle(r, R, color="red")
ax.add_patch(ball_patch)

ball_patch2 = Circle(r2, R2, color="blue")
ax.add_patch(ball_patch2)

ball_patch3 = Circle(r3, R3, color="green")
ax.add_patch(ball_patch3)

for _ in range(500):

    Fg = np.array([0, -m * g])
    dv = Fg / m

    v += dv * dt
    r += v * dt

    Fg2 = np.array([0, -m2 * g])
    dv2 = Fg2 / m2

    v2 += dv2 * dt
    r2 += v2 * dt

    Fg3 = np.array([0, -m3 * g])
    dv3 = Fg3 / m3

    v3 += dv3 * dt
    r3 += v3 * dt


    ball_patch.center = r

    ball_patch2.center = r2

    ball_patch3.center = r3

    plt.pause(0.001)

plt.show()
