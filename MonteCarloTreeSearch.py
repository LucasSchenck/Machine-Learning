import random
import math
import numpy as np

# An implementation of the Monte Carlo Tree Search algorithm for a Tic-Tac-Toe AI

class Node:

    def __init__(self, tiles, parent, move_added, current_player=0):
        self.tiles = tiles
        self.parent = parent
        self.move_added = move_added
        self.current_player = current_player
        self.children = []
        self.simulations = 0
        self.wins = 0
        self.draws = 0

    def get_legal_moves(self):
        # Takes a sequence of game states representing the full game history, and
        # returns the full list of legal moves for the current player.
        legal_moves = []
        for value in range(9):
            if self.tiles[value] is -1:
                legal_moves.append(value)

        return legal_moves

    def get_new_moves(self):
        # Returns a list of moves for which nodes have not been created yet
        new_moves = self.get_legal_moves()
        for i in self.get_legal_moves():
            for j in self.children:
                if j.move_added == i:
                    new_moves.remove(i)

        return new_moves

    def winner(self):
        # Takes a sequence of game states representing the full game history.
        # If the game is won, returns the player number.
        # If the game is still ongoing, returns -1.
        # If the game is tied, returns 0;

        for i in range(3):
            if self.tiles[3*i] == self.tiles[3*i + 1] == self.tiles[3*i + 2] and self.tiles[3*i] != -1:
                return 1 if self.tiles[3*i] == 1 else 2
            elif self.tiles[i] == self.tiles[i + 3] == self.tiles[i + 6] and self.tiles[i] != -1:
                return 1 if self.tiles[i] == 1 else 2

        if self.tiles[0] == self.tiles[4] == self.tiles[8] and self.tiles[0] != -1:
            return 1 if self.tiles[0] == 1 else 2

        if self.tiles[2] == self.tiles[4] == self.tiles[6] and self.tiles[2] != -1:
            return 1 if self.tiles[2] == 1 else 2

        for i in range(9):
            if self.tiles[i] == -1:
                break
            if i == 8:
                return 0

        return -1


class MonteCarloTree:

    def __init__(self, exploration_rate):
        self.root = Node([-1, -1, -1, -1, -1, -1, -1, -1, -1], None, None, 1)
        self.new_node_created = False
        self.exploration_rate = exploration_rate

    def get_play(self, move):
        # Causes the AI to calculate the best move from the current game state node and
        # returns the node corresponding to the move.
        # If no move has been taken so far returns a new node chosen randomly from all legal moves
        if not move.get_legal_moves():
            self.new_node_created = True
            return move

        if random.randint(0, 10) < (self.exploration_rate * 10) or not move.children:
            # print("Selecting random move.")
            if not move.get_new_moves():
                return random.choice(move.children)
            else:
                new_moves = move.get_new_moves()
                new_move_tile = new_moves[random.randint(0, len(new_moves)-1)]
                new_move_tiles = []
                for i in range(9):
                    new_move_tiles.append(move.tiles[i])
                new_move_tiles[new_move_tile] = move.current_player
                new_move_player = 1 if move.current_player == 2 else 2
                new_move = Node(new_move_tiles, move, new_move_tile, new_move_player)
                move.children.append(new_move)
                self.new_node_created = True
                # print("Created new node for player " + str(move.current_player) +
                #       " with tiles: " + str(new_move_tiles) + ", and move " + str(new_move_tile))
                return new_move

        max_value = float('-inf')
        max_node = None
        for i in move.children:
            if move.current_player == 1:
                temp = i.wins/i.simulations + self.exploration_rate * math.sqrt(math.log(move.simulations)/i.simulations)
            else:
                temp = (i.wins + i.draws) / i.simulations + self.exploration_rate * math.sqrt(math.log(move.simulations) / i.simulations)
            if temp > max_value:
                max_value = temp
                max_node = i

        return max_node

    def run_episode(self, move):
        # Runs one episode including selection, expansion, simulation and backpropagation based on the given move.

        # Selection and expansion
        # print("Selecting move...")
        new_move = self.get_play(move)
        while not self.new_node_created:
            # print("Going deeper...")
            new_move = self.get_play(new_move)
        self.new_node_created = False

        # Check if game is over and backpropagate result if it is over
        if new_move.winner() != -1:
            # print("Is end node. Backpropagating...")
            if new_move.winner() == 1:
                self.backpropagate(new_move, 1)
            elif new_move.winner() == 2:
                self.backpropagate(new_move, 2)
            elif new_move.winner() == 0:
                self.backpropagate(new_move, 0)
            return

        # Simulate result with random moves and backpropagate result
        winner = self.run_simulation(new_move)
        # print("Is not end node. Winner of simulation is " + str(winner) + ". Backpropagating...")
        if winner == 1:
            self.backpropagate(new_move, 1)
        elif winner == 2:
            self.backpropagate(new_move, 2)
        elif winner == 0:
            self.backpropagate(new_move, 0)
        return

    def run_simulation(self, move):
        # print("Ended on tiles: " + str(move.tiles) + ". Running simulation...")
        tiles = []
        for i in range(9):
            tiles.append(move.tiles[i])
        legal_moves = move.get_legal_moves()
        tiles[random.choice(legal_moves)] = move.current_player
        player = move.current_player
        while self.winner(tiles) == -1:
            player = 1 if player == 2 else 2
            legal_moves = self.get_legal_moves(tiles)
            tiles[random.choice(legal_moves)] = player

        # print("Finished simulation. Winner is: " + str(self.winner(tiles)) + " with tiles " + str(tiles) +
        #       ". move.tiles == " + str(move.tiles))
        return self.winner(tiles)

    def backpropagate(self, move, winner):
        current = move
        while current != self.root:
            if current.current_player != winner:
                if winner == 0:
                    current.draws += 1
                else:
                    current.wins += 1

            current.simulations += 1
            current = current.parent

        self.root.simulations += 1

    @staticmethod
    def winner(tiles):
        # Takes a list of tiles and:
        # If the game is won, returns the player number.
        # If the game is still ongoing, returns -1.
        # If the game is tied, returns 0;

        for i in range(3):
            if tiles[3*i] == tiles[3*i + 1] == tiles[3*i + 2] and tiles[3*i] != -1:
                return 1 if tiles[3*i] == 1 else 2
            elif tiles[i] == tiles[i + 3] == tiles[i + 6] and tiles[i] != -1:
                return 1 if tiles[i] == 1 else 2

        if tiles[0] == tiles[4] == tiles[8] and tiles[0] != -1:
            return 1 if tiles[0] == 1 else 2

        if tiles[2] == tiles[4] == tiles[6] and tiles[2] != -1:
            return 1 if tiles[2] == 1 else 2

        for i in range(9):
            if tiles[i] == -1:
                break
            if i == 8:
                return 0

        return -1

    @staticmethod
    def get_legal_moves(tiles):
        # Takes a list of tiles and
        # returns the full list of legal moves for the current player.
        legal_moves = []
        for value in range(9):
            if tiles[value] is -1:
                legal_moves.append(value)

        return legal_moves

    def get_best_move(self, tiles):
        if self.winner(tiles) != -1:
            print("Game is already over. There is no best move.")
            return -1;

        current = self.root
        while not np.array_equal(current.tiles, tiles):
            for i in range(9):
                if current.tiles[i] == -1 and tiles[i] == current.current_player:
                    for child in current.children:
                        if child.move_added == i:
                            current = child
                            break
                else:
                    continue
                break

        return self.get_play(current).move_added


tree = MonteCarloTree(math.sqrt(2))
for i in range(0, 10_000_000):
    tree.run_episode(tree.root)
    tree.exploration_rate = tree.exploration_rate*0.9999
    if i % 50_000 == 0:
        print("Episode " + str(i) + ": root wins = " + str(tree.root.wins) + ", simulations = " + str(tree.root.simulations))
        for child in tree.root.children:
            print("move " + str(child.move_added) + " wins = " + str(child.wins) + ", simulations = " + str(child.simulations))

tree.exploration_rate = 0
play = tree.root
while tree.winner(play.tiles) == -1:
    print(str(play.tiles) + " Wins: " + str(play.wins) + ", Simulations: " + str(play.simulations))
    play = tree.get_play(play)
print(str(play.tiles))
print("Best play in situation [1, 1, -1, -1, 2, -1, -1, -1, -1]: " + str(tree.get_best_move([1, 1, -1, -1, 2, -1, -1, -1, -1])))
print("Best play in situation [1, 1, -1, 2, -1, -1, 2, -1, -1]: " + str(tree.get_best_move([1, 1, -1, 2, -1, -1, 2, -1, -1])))
print("Best play in situation [1, 1, 2, 2, -1, -1, -1, -1, -1]: " + str(tree.get_best_move([1, 1, 2, 2, -1, -1, -1, -1, -1])))
print("Best play in situation [1, 1, 2, 2, 1, -1, -1, -1, -1]: " + str(tree.get_best_move([1, 1, 2, 2, 1, -1, -1, -1, -1])))
print("Best play in situation [1, 1, 2, 2, 1, -1, -1, 2, -1]: " + str(tree.get_best_move([1, 1, 2, 2, 1, -1, -1, 2, -1])))
print("Best play in situation [1, 1, 2, 2, 2, -1, -1, -1, 1]: " + str(tree.get_best_move([1, 1, 2, 2, 2, -1, -1, -1, 1])))
