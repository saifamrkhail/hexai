import numpy as np
import torch
import torch.nn.functional as F
from random import choices, shuffle
from MCTS import MCTS
import Model as Model
import hex_engine as hex_engine
import agent as AI

BOARD_SIZE=3 # ToDo: When algorithm is stable change board size here to 7 for 7x7. Test on 3 (3x3) for quick iterative development


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
        for iteration in range(self.args['num_iterations']):
            print(iteration, "### Running Iteration")
            memory = []
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                print(selfPlay_iteration, "## Playing Episode/Rollout")
                memory += self.selfPlay()
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                print(epoch, "# Epoch/Forward Pass")
                self.train(memory)

        torch.save(self.model.state_dict(), "models/model.pt")
        torch.save(self.optimizer.state_dict(), "models/optimizer.pt")

        #game.machine_vs_machine(machine1=AI.machine, machine2=AI.machine)
            
            # # ToDo:
            # if 0 == (iteration % 10):
            #     torch.save(self.model.state_dict(), "model_"+ iteration + "_.pt")
            #     torch.save(self.optimizer.state_dict(), "optimizer_"+ iteration +"_.pt")
            #     if 0 == (iteration % 100):
            #         torch.save(self.model.state_dict(), "modelCheckpoint_"+ iteration/10 + "_.pt")
            #        # ToDo: let modelCheckpoint train against last 10 model iterations and check if better

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = hex_engine.hexPosition(size=BOARD_SIZE) 
    model = Model.ResNet(game, 4, 64, device) # ToDo: "64" needs to be changed with board size i think
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 2,
        'num_selfPlay_iterations': 10,
        'num_epochs': 3,
        'batch_size': 50,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZero(model, game, optimizer, args)
    alphaZero.learn()
