import numpy as np
import torch
import Model as Model
import hex_engine as hex_engine

BOARD_SIZE=5

def machine(board, action_set):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        game = hex_engine.hexPosition(size=BOARD_SIZE) 
        chosen_action_set = game.get_action_space()
        loaded_model = Model.ResNet(game, 4, 64, device)
        loaded_model.load_state_dict(torch.load("models/AZagent.pt"))

        board = np.array(board)
        board = np.stack((board == 1, board == 0, board == -1)).astype(np.float32)
        state = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        loaded_model.eval()
        out_policy, _ = loaded_model(state)
        action_index = np.argmax(out_policy.detach().numpy())
        
        while chosen_action_set[action_index] not in action_set:
            out_policy[0][action_index] = -1
            action_index = np.argmax(out_policy.detach().numpy())

        print("valid action chosen: ", chosen_action_set[action_index])
        return chosen_action_set[action_index]