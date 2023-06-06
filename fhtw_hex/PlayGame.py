import Model as Model
import hex_engine as hex_engine
import RunAgent as AI

BOARD_SIZE=3
game = hex_engine.hexPosition(size=BOARD_SIZE)

#play against the machine
game.human_vs_machine(human_player=1, machine=AI.machine)