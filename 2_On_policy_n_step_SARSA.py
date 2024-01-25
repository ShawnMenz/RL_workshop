import numpy as np

# environment
MAP = [["S", "F", "H"], ["H", "F", "G"]]
map_row = len(MAP)
map_col = len(MAP[0])
num_state = map_row * map_col

# action
# RIGHT = 0, DOWN = 1
ACTION = [0, 1]
num_action = len(ACTION)

# Q table
Q_table = np.zeros((num_state, num_action))

# hyperparameters
lr = 0.1
discount = 0.9
episodes = 1000
n = 4


# helper function
def action_to_arrow(action):
    if action == 0:
        return "→"
    elif action == 1:
        return "↓"


for episode in range(episodes):
    # current state
    current_state = MAP[0][0]
    current_state_index_x = 0
    current_state_index_y = 0

    # epsilon config
    if episode < int(episodes / 2):
        epsilon = 0.5
    else:
        epsilon = 0.1

    T = float("inf")
    t = 0

    states = [(current_state_index_x, current_state_index_y)]
    actions = []
    rewards = [0]

    while True:
        if t < T:
            # action choice (epsilon greedy)
            if np.random.rand() < epsilon:
                # exploration
                action = np.random.choice(ACTION)
            else:
                # exploitation
                action = np.argmax(
                    Q_table[current_state_index_x * map_col + current_state_index_y]
                )

            actions.append(action)

            # next state
            next_state_index_x = current_state_index_x
            next_state_index_y = current_state_index_y
            if action == 0:
                # RIGHT
                if current_state_index_y < map_col - 1:
                    next_state_index_y = current_state_index_y + 1
            elif action == 1:
                # DOWN
                if current_state_index_x < map_row - 1:
                    next_state_index_x = current_state_index_x + 1

            next_state = MAP[next_state_index_x][next_state_index_y]
            states.append((next_state_index_x, next_state_index_y))

            # reward
            if next_state == "H":
                rewards.append(-1)
            elif next_state == "G":
                rewards.append(1)
            else:
                rewards.append(0)

            # update current state
            current_state = next_state
            current_state_index_x = next_state_index_x
            current_state_index_y = next_state_index_y

            if current_state == "G" or current_state == "H":
                T = t + 1
            else:
                if np.random.rand() < epsilon:
                    # exploration
                    action = np.random.choice(ACTION)
                else:
                    # exploitation
                    action = np.argmax(
                        Q_table[current_state_index_x * map_col + current_state_index_y]
                    )
                actions.append(action)

        tau = t - n + 1
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += pow(discount, i - tau - 1) * rewards[i]

            if tau + n < T:
                G = (
                    G
                    + pow(discount, n)
                    * Q_table[states[tau + n][0] * map_col + states[tau + n][1]][
                        actions[tau + n]
                    ]
                )
            Q_table[states[tau][0] * map_col + states[tau][1]][actions[tau]] += lr * (
                G - Q_table[states[tau][0] * map_col + states[tau][1]][actions[tau]]
            )

        t += 1

        if tau == T - 1:
            break

print("Q_table: \n", Q_table)
print("Optimal Path:")
for i in range(map_row):
    for j in range(map_col):
        print(action_to_arrow(np.argmax(Q_table[i * map_col + j])), end=" ")
    print()
