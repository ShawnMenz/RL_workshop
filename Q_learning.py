import numpy as np
import matplotlib.pyplot as plt

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
episodes = 10


# helper function
def action_to_arrow(action):
    if action == 0:
        return "→"
    elif action == 1:
        return "↓"


# Q learning
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

    step = 1
    while True:
        # create plot
        fig, ax = plt.subplots()
        x_labels = ["Right", "Down"]
        y_labels = ["S(S0)", "F(S1)", "H(S2)", "H(S3)", "F(S4)", "G(S5)"]
        ax.axis("off")

        title_data = ""
        title_data += "Episode: " + str(episode + 1) + ", "
        title_data += "Step: " + str(step) + ", "
        step += 1
        title_data += (
            "Current State: "
            + str(current_state)
            + "(S"
            + str(current_state_index_x * map_col + current_state_index_y)
            + ")"
            + ", "
        )

        # action choice (epsilon greedy)
        if np.random.rand() < epsilon:
            # exploration
            action = np.random.choice(num_action)
        else:
            # exploitation
            action = np.argmax(
                Q_table[current_state_index_x * map_col + current_state_index_y]
            )

        # next state
        next_state_index_x = current_state_index_x
        next_state_index_y = current_state_index_y

        # take action and get next states
        if action == 0:
            # RIGHT
            if current_state_index_y < map_col - 1:
                next_state_index_y = current_state_index_y + 1
                title_data += "Action: Right"
        elif action == 1:
            # DOWN
            if current_state_index_x < map_row - 1:
                next_state_index_x = current_state_index_x + 1
                title_data += "Action: Down"

        # reward
        if MAP[next_state_index_x][next_state_index_y] == "H":
            # hole
            reward = -1
        elif MAP[next_state_index_x][next_state_index_y] == "G":
            # goal
            reward = 1
        else:
            # ice and all other states
            reward = 0

        # update Q table
        current_state_in_table = current_state_index_x * map_col + current_state_index_y
        next_state_in_table = next_state_index_x * map_col + next_state_index_y
        Q_table[current_state_in_table, action] = Q_table[
            current_state_in_table, action
        ] + lr * (
            reward
            + discount * np.max(Q_table[next_state_in_table])
            - Q_table[current_state_in_table, action]
        )

        # update current state
        current_state_index_x = next_state_index_x
        current_state_index_y = next_state_index_y
        current_state = MAP[current_state_index_x][current_state_index_y]

        plt.title(title_data, fontsize=8)
        table = ax.table(
            cellText=Q_table, colLabels=x_labels, rowLabels=y_labels, loc="center"
        )
        plt.pause(0.5)
        plt.show()

        # terminal states
        if (
            MAP[current_state_index_x][current_state_index_y] == "G"
            or MAP[current_state_index_x][current_state_index_y] == "H"
        ):
            break


print("Q_table: \n", Q_table)
print("Optimal Path:")
for i in range(map_row):
    for j in range(map_col):
        print(action_to_arrow(np.argmax(Q_table[i * map_col + j])), end=" ")
    print()
