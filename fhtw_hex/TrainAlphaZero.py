import numpy as np
import torch
import torch.nn.functional as F
from random import shuffle
from MCTS import MCTS
import Model as Model
import hex_engine as hex_engine

BOARD_SIZE=5 #Size of the board for TRAINING AND PLAYING
CYCLES=10 #How many models are being generated (with "AZagent.pt" being the final model)

class AlphaZero:
    def __init__(self, model, game, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        self.game.reset()
        player = 1

        while True:
            if self.game.player == -1:
                self.game.board = self.game.recode_black_as_white()
                self.game.player = 1
                player = -1

            action_probs = self.mcts.search()
            state = np.array(self.game.board)
            memory.append((np.stack((state == 1, state == 0, state == -1)).astype(np.float32), action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= sum(temperature_action_probs)
            action = self.game.scalar_to_coordinates(np.random.choice(self.game.size ** 2, p=temperature_action_probs))

            self.game.moove(action)

            value, is_terminal = self.game.winner, self.game.winner != 0 or len(self.game.get_action_space()) == 0

            if is_terminal:
                returnMemory = []
                for hist_game_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    returnMemory.append((
                        hist_game_state,
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            if player == -1:
                self.game.board = self.game.recode_black_as_white()
                self.game.player = 1
                player = 1

    def train(self, memory):
        if not memory:
            return

        shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def learn(self):

        for cycle in range(self.args['num_cycles']):
            print("### Running Training Cycle {} ###".format(cycle+1))
            memory = []

            self.model.eval()
            print("## Playing Episodes / Generating Rollouts ##")
            for episodes in range(self.args['num_episodes']):
                print("# Episode {} #".format(episodes+1))
                memory += self.selfPlay()

            self.model.train()
            print("## Forward Passes / Epochs ##")
            for epoch in range(self.args['num_epochs']):
                print("# Epoch/Forward Pass {} #" .format(epoch+1))
                self.train(memory)

            if 0 == (cycle % 10): # muss nicht 10 sein
                print("### Saving Checkpoint model {} ###" .format(cycle))
                torch.save(self.model.state_dict(), "models/model_"+str(cycle)+".pt")
                # ToDo: 

                # Let modelCheckpoint train against last previous model cycles and

                # if better: print("### Model improved ###") and continue training cycles.
                # and output improvement statistics + graphs (e.g. winrate)

                # if not better: reject model, 
                # "cycle -= 10~, model_rejected_cnt+=1", " *adjust exploratory hyper parameter* ", 
                # load last/older checkpoint model self.model.load_state_dict(torch.load("models/model_"+str(cycle)+"_Checkpoint.pt")
                # and resume/repeat training/overwrite last 10~ models --> continue.

                # If not better any arbitrary amount of times or finishes all cycles, stop training and save last model.
                # if model_rejected_cnt > 5:
                #     print("### Finishing training as model fails to improve winrate ###")
                #     break

            else:
                torch.save(self.model.state_dict(), "models/model_"+str(cycle)+".pt")
        
        torch.save(self.model.state_dict(), "models/AZagent.pt")
        
        print("### Training finished after {} cycles - Agent saved as models/AZagent.pt ###" .format(cycle))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = hex_engine.hexPosition(size=BOARD_SIZE) 
    model = Model.ResNet(game, 4, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    #finale parameter zum trainieren
    #80 iterations
    #100 episoden
    #25 MCTS (epochen)

    args = {
        'C': 2,
        'num_searches': 60,
        'num_cycles': CYCLES,
        'num_episodes': 50,
        'num_epochs': 13,
        'batch_size': 50, # sollte mehr als die Anzahl der Episoden entsprechen (= num_episodes)
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }


    alphaZero = AlphaZero(model, game, optimizer, args)
    alphaZero.learn()