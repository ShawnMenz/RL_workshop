import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


# Path config
IMG_PATH = "./img/"
EPISODE_PATH = "./img/ql/"
OPTIMAL_GRID_PATH = "./img/q_learning_optimal_path.png"

for path in [IMG_PATH, EPISODE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
    

# helper function
def action_to_arrow(action):
    if action == 0:
        return "→"
    elif action == 1:
        return "↓"


def flatten_list(list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def plot_optimal_path(Q_table):
    current_state = (0, 0)  # Start state
    optimal_path = [current_state]
    actions = []

    while MAP[current_state[0]][current_state[1]] != "G":
        action = np.argmax(Q_table[current_state[0] * map_col + current_state[1]])
        actions.append(action_to_arrow(action))
        if action == 0 and current_state[1] < map_col - 1:
            current_state = (current_state[0], current_state[1] + 1)
        elif action == 1 and current_state[0] < map_row - 1:
            current_state = (current_state[0] + 1, current_state[1])
        optimal_path.append(current_state)

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, map_col - 0.5)
    ax.set_ylim(-0.5, map_row - 0.5)

    # Plotting the grid and the optimal path
    for i, ((x, y), action) in enumerate(zip(optimal_path, actions + [""])):
        ax.text(
            y,
            map_row - 1 - x,
            MAP[x][y] + "\n" + action,
            ha="center",
            va="center",
            fontsize=20,
            color="blue",
        )
        if i < len(actions):
            next_state = optimal_path[i + 1]
            ax.annotate(
                "",
                xy=(next_state[1], map_row - 1 - next_state[0]),
                xytext=(y, map_row - 1 - x),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="red"),
            )

    ax.set_xticks(np.arange(-0.5, map_col, 1))
    ax.set_yticks(np.arange(-0.5, map_row, 1))
    ax.grid(which="both")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title("Optimal Path")
    plt.savefig(f"{OPTIMAL_GRID_PATH}")
    plt.close()


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

    step = 0
    while True:
        meta_data = "episode: " + str(episode)
        step += 1
        meta_data += ", step: " + str(step)
        meta_data += (
            ", current state: S"
            + str(current_state_index_x * map_col + current_state_index_y)
            + " - "
            + MAP[current_state_index_x][current_state_index_y]
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
                meta_data += ", Action: Right"
        elif action == 1:
            # DOWN
            if current_state_index_x < map_row - 1:
                next_state_index_x = current_state_index_x + 1
                meta_data += ", Action: Down"

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

        # plot and save image
        # create a new figure and set the size
        plt.figure(figsize=(10, 4))
        # create a heatmap using Seaborn with numbers rounded to two decimal places
        ax = sns.heatmap(
            Q_table,
            cmap="hot",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 16},
            vmax=1,
            vmin=-1,
        )
        # ensure entire boundaries are visible
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_xticklabels(["Right", "Down"])
        plt.xticks(fontsize=20)
        flat_list = flatten_list(MAP)
        ax.set_yticklabels(
            ["S" + str(i) + " - " + element for i, element in enumerate(flat_list)]
        )
        plt.yticks(fontsize=20, rotation=0)
        plt.title(meta_data, fontsize=16)
        plt.savefig(f"{EPISODE_PATH}episode-" + str(episode+1) + " step-" + str(step) + ".png")
        # plt.show()
        plt.close()

        # terminal states
        if (
            MAP[current_state_index_x][current_state_index_y] == "G"
            or MAP[current_state_index_x][current_state_index_y] == "H"
        ):
            break


plot_optimal_path(Q_table)
