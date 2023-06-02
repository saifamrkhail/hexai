#make sure that the module is located somewhere where your Python system looks for packages
import sys
sys.path.append("/home/sharwin/Desktop/")

import numpy as np
import torch
import torch.nn.functional as F
from random import choices, shuffle
from MCTS import MCTS
import Model as Model
import hex_engine as hex_engine
import agent as AI

BOARD_SIZE=3

#initializing a game object
game = hex_engine.hexPosition(size=BOARD_SIZE)

#play against the machine
game.human_vs_machine(human_player=1, machine=AI.machine)

#let machine play against random
game.machine_vs_machine(machine1=AI.machine, machine2=None)
