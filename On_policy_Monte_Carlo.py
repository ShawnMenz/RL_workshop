import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# environment
# S: start, F: frozen, H: hole, G: goal
#
# map:
# S F H
# H F G
MAP = [["S", "F", "H"], ["H", "F", "G"]]
map_row = len(MAP)
map_col = len(MAP[0])
num_state = map_row * map_col

# action
# RIGHT = 0, DOWN = 1
ACTION = [0, 1]
num_action = len(ACTION)

# hyperparameters
episodes = 10
discount = 0.9

# Path config
IMG_PATH = "./img/"
EPISODE_PATH = "./img/op/"
STEP_PATH = "./img/op_grid/"
OPTIMAL_GRID_PATH = "./img/on_policy_optimal_path.png"

for path in [IMG_PATH, EPISODE_PATH, STEP_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


# helper function to convert action to arrow
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


# Function to plot the grid
def plot_grid(current_state, action, next_state, episode, step):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, map_col - 0.5)
    ax.set_ylim(-0.5, map_row - 0.5)

    # Plotting the grid
    for x in range(map_row):
        for y in range(map_col):
            ax.text(
                y, map_row - 1 - x, MAP[x][y], ha="center", va="center", fontsize=20
            )
            if (x, y) == current_state:
                ax.plot(y, map_row - 1 - x, "ro")  # Current state
            if (x, y) == next_state:
                ax.plot(y, map_row - 1 - x, "go")  # Next state

    # Adding the action arrow
    ax.annotate(
        "",
        xy=(next_state[1], map_row - 1 - next_state[0]),
        xytext=(current_state[1], map_row - 1 - current_state[0]),
        arrowprops=dict(arrowstyle="->", lw=2.0),
    )

    ax.set_xticks(np.arange(-0.5, map_col, 1))
    ax.set_yticks(np.arange(-0.5, map_row, 1))
    ax.grid(which="both")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title(f"Episode {episode+1}, Step {step}")
    plt.savefig(f"{STEP_PATH}episode-{episode+1}-step-{step}.png")
    plt.close()


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


# Initialize Q-table and Return_s_a
Q_table = np.zeros((num_state, num_action))
Return_s_a = {}

# Main loop for each episode
for episode in range(episodes):
    steps = []
    # start state
    current_state = MAP[0][0]
    current_state_index_x = 0
    current_state_index_y = 0

    # epsilon config: decayed epsilon greedy (simple implementation)
    if episode < int(episodes / 2):
        epsilon = 0.5
    else:
        epsilon = 0.1

    while True:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(ACTION)
        else:
            action = np.argmax(
                Q_table[current_state_index_x * map_col + current_state_index_y]
            )

        # Calculate next state
        next_state_index_x = current_state_index_x
        next_state_index_y = current_state_index_y

        if action == 0 and current_state_index_y < map_col - 1:
            next_state_index_y += 1
        elif action == 1 and current_state_index_x < map_row - 1:
            next_state_index_x += 1

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

        # Store step
        steps.append(((current_state_index_x, current_state_index_y), action, reward))

        # Plot current step
        plot_grid(
            (current_state_index_x, current_state_index_y),
            action,
            (next_state_index_x, next_state_index_y),
            episode,
            len(steps),
        )

        # Update current state
        current_state_index_x = next_state_index_x
        current_state_index_y = next_state_index_y
        current_state = MAP[current_state_index_x][current_state_index_y]

        # Check for terminal state
        if current_state in ["G", "H"]:
            steps.append(((current_state_index_x, current_state_index_y), action, reward))
            break

    # Process the episode for Q-table update
    G = 0
    for i, t in enumerate(reversed(range(len(steps)))):
        # meta data for Q-table title
        meta_data = "episode: " + str(episode + 1)

        state = steps[t][0]
        current_x = state[0]
        current_y = state[1]
        meta_data += ", step: " + str(i + 1)
        meta_data += (
            ", current state: S"
            + str(current_x * map_col + current_y)
            + " - "
            + MAP[current_x][current_y]
        )

        action = steps[t][1]
        if action == 0:
            meta_data += ", action: RIGHT"
        elif action == 1:
            meta_data += ", action: DOWN"

        reward = steps[t][2]

        G = discount * G + reward
        # unless the S, A in the episode
        if (state, action) not in [(x[0], x[1]) for x in steps[:t]]:
            # append G to Return(S, A)
            if (state, action) not in Return_s_a:
                Return_s_a[(state, action)] = []
            Return_s_a[(state, action)].append(G)
            # avg(Return)
            Q_table[current_x * map_col + current_y, action] = np.mean(
                Return_s_a[(state, action)]
            )

        # Update Q-table heatmap
        plt.figure(figsize=(10, 4))
        ax = sns.heatmap(
            Q_table,
            annot=True,
            cmap="hot",
            fmt=".2f",
            annot_kws={"size": 16},
            vmax=1,
            vmin=-1,
        )
        ax.set_ylim(len(Q_table) + 0.5, -0.5)
        ax.set_xticklabels(["Right", "Down"])
        plt.xticks(fontsize=20)
        flat_list = flatten_list(MAP)
        ax.set_yticklabels(
            ["S" + str(i) + " - " + element for i, element in enumerate(flat_list)]
        )
        plt.yticks(fontsize=20, rotation=0)
        plt.title(meta_data, fontsize=16)
        plt.savefig(f"{EPISODE_PATH}episode-{episode + 1}-step-{i+1}.png")
        plt.close()


plot_optimal_path(Q_table)
