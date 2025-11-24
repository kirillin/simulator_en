import numpy as np

from physics_engine import PhysicsEngine
from renderer import Renderer
from world import World

from objects import Model

from dynamics.aba_solver import ABASolver

def main():

    fd_solver = ABASolver()

    physics = PhysicsEngine(fd_solver)
    renderer = Renderer()

    world = World(physics, renderer)

    robot = Model(5)

    world.add_object(robot)
    world.run(5000)

if __name__=="__main__":
    main()
