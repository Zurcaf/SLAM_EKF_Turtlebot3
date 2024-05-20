from utils import vehicles, environment
import numpy as np
import pygame

if __name__=='__main__':

    # Initialize pygame
    pygame.init()

    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2])
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Initialize and display environment
    env = environment.Environment(map_image_path="./maps/map_blank.png")

    running = True
    u = np.array([0.,0.]) # Controls
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        robot.move_step(u,dt) # Integrate EOMs forward, i.e., move robot
        env.show_map() # Re-blit map
        env.show_robot(robot) # Re-blit robot
        pygame.display.update() # Update display
    