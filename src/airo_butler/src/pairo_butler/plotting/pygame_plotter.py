from ast import List
from typing import Tuple
from PIL import Image
import cv2
import numpy as np

import pygame

import rospy


class PygameWindow:
    def __init__(self, caption: str, size: Tuple[int, int] = (512, 512)) -> None:
        pygame.init()
        self.window_size = size
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(caption)

        self.__events: List[pygame.Event] = []

    @property
    def events(self):
        return self.__events

    def process_events(self):
        self.__events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rospy.signal_shutdown("Pygame window closes")
            else:
                self.__events.append(event)

    def imshow(self, image: Image):
        self.process_events()
        image = np.array(image).transpose(1, 0, 2)
        image = cv2.resize(image, self.window_size[::-1])
        frame_surface = pygame.surfarray.make_surface(image)
        self.screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
