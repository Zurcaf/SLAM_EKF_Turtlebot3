'''
Script for environment classes and functions
'''
import numpy as np
import sys
import pygame
import pygame.gfxdraw
import pdb

class Environment:
    METER_PER_PIXEL = 0.015
    # METER_PER_PIXEL = 0.025 # Conversion from meters to pixels
    # METER_PER_PIXEL = 0.035 # Conversion from meters to pixels
    def __init__(self,map_image_path):
        self.map_image = pygame.image.load(map_image_path) # Load map image
        pygame.display.set_caption("map")
        self.map = pygame.display.set_mode((self.map_image.get_width(),self.map_image.get_height()))
        self.show_map() # Blit map image onto display
        # Preset colors
        self.black = (0, 0, 0)
        self.grey = (70, 70, 70)
        self.dark_grey= (20,20,20)
        self.blue = (0, 0, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.white = (255, 255, 255)

    def pixel2position(self,pixel):
        '''
        Convert pixel into position
        '''
        posx = pixel[0]*self.METER_PER_PIXEL
        posy = (self.map.get_height()-pixel[1])*self.METER_PER_PIXEL
        return np.array([posx,posy])

    def position2pixel(self,position):
        '''
        Convert position into pixel
        '''
        pixelx = int(position[0]/self.METER_PER_PIXEL)
        pixely = int(self.map.get_height() - position[1]/self.METER_PER_PIXEL)
        return np.array([pixelx,pixely])
    
    def dist2pixellen(self,dist):
        return int(dist/self.METER_PER_PIXEL)

    def show_map(self):
        '''
        Blit map onto display
        '''
        self.map.blit(self.map_image,(0,0))

    def show_robot(self,robot):
        '''
        Blit robot onto display
        '''
        corners = robot.get_corners()
        pixels = [self.position2pixel(corner) for corner in corners]
        # pygame.draw.polygon(self.map,self.grey,pixels) # filled in polygon, not anti-aliased
        pygame.gfxdraw.aapolygon(self.map,pixels,self.dark_grey) # Anti-aliased outline
        pygame.gfxdraw.filled_polygon(self.map,pixels,self.grey)

    def get_pygame_surface(self):
        return self.map

if __name__=='__main__':
    '''
    Test display of map
    '''
    pygame.init()

    map_path = "./maps/map_blank.png"

    env = Environment(map_path)

    running = True
    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
        env.show_map()
        pygame.display.update()