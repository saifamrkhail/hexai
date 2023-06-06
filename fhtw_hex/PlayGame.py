import Model as Model
import hex_engine as hex_engine
from StatisticGen import machine as AI
import TrainAlphaZero as taz

game = hex_engine.hexPosition(size=taz.BOARD_SIZE)

#play against the machine
#game.human_vs_machine(human_player=1, machine=lambda board, action_set: AI(board, action_set, modelPath = "models/model_7.pt"))
#game.human_vs_machine(human_player=-1, machine=lambda board, action_set: AI(board, action_set, modelPath = "models/model_7.pt"))


#TEST
wins=0
losses=0
for i in range(100):
    game.machine_vs_machine(machine1=lambda board, action_set: AI(board, action_set, modelPath = "models/model_7.pt"), machine2=None)
    
    if(game.winner == 1):
        wins += 1
    elif(game.winner == -1):
        losses += 1
    
print("Wins=", wins," | Losses=", losses)