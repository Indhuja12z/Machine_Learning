import numpy as np
import random
import tkinter as tk
from collections import defaultdict
# Q-Learning Agent

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, board):
        return str(board)

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        q_values = self.q_table[state]
        return max(available_moves, key=lambda x: q_values[x])

    def update(self, state, action, reward, next_state, done):
        max_future_q = np.max(self.q_table[next_state]) if not done else 0
        current_q = self.q_table[state][action]

        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )

# Game Logic
def check_winner(board):
    wins = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for w in wins:
        if board[w[0]] == board[w[1]] == board[w[2]] != 0:
            return board[w[0]]
    if 0 not in board:
        return 0  # draw
    return None

def available_moves(board):
    return [i for i in range(9) if board[i] == 0]

# Training

def train(agent, episodes=5000):
    for _ in range(episodes):
        board = [0]*9
        state = agent.get_state(board)

        while True:
            moves = available_moves(board)
            action = agent.choose_action(state, moves)
            board[action] = 1  # AI move

            winner = check_winner(board)
            next_state = agent.get_state(board)

            if winner is not None:
                reward = 1 if winner == 1 else 0
                agent.update(state, action, reward, next_state, True)
                break

            # Opponent random move
            opp_move = random.choice(available_moves(board))
            board[opp_move] = -1

            winner = check_winner(board)
            next_state = agent.get_state(board)

            if winner is not None:
                reward = -1 if winner == -1 else 0
                agent.update(state, action, reward, next_state, True)
                break

            agent.update(state, action, 0, next_state, False)
            state = next_state

# UI using Tkinter
class TicTacToeUI:
    def __init__(self, agent):
        self.agent = agent
        self.board = [0]*9

        self.window = tk.Tk()
        self.window.title("RL Tic Tac Toe")

        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.window, text="", font=('Arial', 20),
                            width=5, height=2,
                            command=lambda i=i: self.player_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

    def player_move(self, index):
        if self.board[index] != 0:
            return

        self.board[index] = -1
        self.buttons[index].config(text="X")

        if self.check_end():
            return

        self.ai_move()

    def ai_move(self):
        state = self.agent.get_state(self.board)
        moves = available_moves(self.board)
        action = self.agent.choose_action(state, moves)

        self.board[action] = 1
        self.buttons[action].config(text="O")

        self.check_end()

    def check_end(self):
        winner = check_winner(self.board)
        if winner is not None:
            if winner == 1:
                self.show_result("AI Wins!")
            elif winner == -1:
                self.show_result("You Win!")
            else:
                self.show_result("Draw!")
            return True
        return False

    def show_result(self, msg):
        for b in self.buttons:
            b.config(state="disabled")
        label = tk.Label(self.window, text=msg, font=('Arial', 16))
        label.grid(row=3, column=0, columnspan=3)

    def run(self):
        self.window.mainloop()

# Run Everything

agent = QLearningAgent()
print("Training AI...")
train(agent, episodes=10000)
print("Training complete!")

game = TicTacToeUI(agent)
game.run()
