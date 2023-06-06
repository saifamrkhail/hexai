import numpy as np
import torch
import Model as Model
import hex_engine as hex_engine
import TrainAlphaZero as taz

def statistics():

    global modelPath
    results=[]
    game=hex_engine.hexPosition(size=taz.BOARD_SIZE) 
    for i in range(0,taz.CYCLES):
        modelPath = "models/model_"+ str(i) +".pt"
        print("######### MODEL = #", i)
        wins=0
        losses=0

        for j in range(100):
            game.machine_vs_machine(machine1=lambda board, action_set: machine(board, action_set, modelPath), machine2=None)
            if(game.winner == 1):
                wins += 1
            elif(game.winner == -1):
                losses += 1

        results.append([str(i)+"# Model",wins,losses])

    print("Results:\n",results)

def machine(board, action_set, modelPath="models/AZagent.pt"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = hex_engine.hexPosition(size=taz.BOARD_SIZE) 
    chosen_action_set = game.get_action_space()
    loaded_model = Model.ResNet(game, 4, 64, device)
    
    loaded_model.load_state_dict(torch.load(modelPath))

    board = np.array(board)
    board = np.stack((board == 1, board == 0, board == -1)).astype(np.float32)
    state = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
    loaded_model.eval()
    out_policy, _ = loaded_model(state)
    action_index = np.argmax(out_policy.detach().numpy())

    max_attempts = taz.BOARD_SIZE ** 2
    attempts = 0

    while chosen_action_set[action_index] not in action_set and attempts < max_attempts:
        out_policy[0][action_index] = -99
        action_index = np.argmax(out_policy.detach().numpy())
        attempts += 1

    if chosen_action_set[action_index] not in action_set:
        # No valid actions found
        return None
    
    return chosen_action_set[action_index]



if __name__ == "__main__":
    statistics()