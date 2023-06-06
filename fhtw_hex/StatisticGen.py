import Model as Model
import hex_engine as hex_engine
import RunAgent as AI
import TrainAlphaZero as taz

MODEL_PATH="models/AZagent.pt"

game = hex_engine.hexPosition(size=taz.BOARD_SIZE)

models = []
#models = loaded_model.load_state_dict(torch.load("models/models_.pt"))

#play against the machine
whiteWins=0
blackWins=0
draw=0

while i in range(taz.CYCLES):
    models = loaded_model.load_state_dict(torch.load("models/models_.pt"))
    for i in range(100):
        game.machine_vs_machine(machine1=AI.machine, machine2=None)
        if(game.winner == 1):
            whiteWins+=1
        elif(game.winner == -1):
            blackWins+=1
        else:
            draw+=1


print("White wins: ", whiteWins)
print("Black wins: ", blackWins)
print("Draws: ", draw)
