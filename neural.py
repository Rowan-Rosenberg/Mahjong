"""
    Defines neural networks for the mahjong environment
    Written by Rowan Rosenberg 2025
"""

import random

class Agent():

    def __intit__(self,models):

        self.models = models

    def choose(self, encoding, options, player_num):

        return random.choice(options)
