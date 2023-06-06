import Model as Model
import hex_engine as hex_engine
import RunAgent as AI

BOARD_SIZE=3
game = hex_engine.hexPosition(size=BOARD_SIZE)

models = []
#models = loaded_model.load_state_dict(torch.load("models/AZagent.pt"))

#play against the machine
whiteWins=0
blackWins=0
draw=0

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
