import numpy as np
import sys
from environment import MountainCar
import matplotlib.pyplot as plt


def main(args):
    if len(sys.argv) < 9:
        print("Please give mode, weight_out file name, return_out file name, episodes, max_iterations, "
              "epsilon, gamma, and learning_rate respectively in commandline arguments")
    mode = sys.argv[1]
    weight_out_file = sys.argv[2]
    return_out_file = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    # initialize environment
    mc = MountainCar(mode=mode)
    action_space = mc.action_space
    state_space = mc.state_space

    # initialize weights and bias
    weights = np.zeros((state_space, action_space))
    bias = 0

    return_rewards = []
    avg_rewards = []
    for i in range(episodes):
        state = mc.reset()
        done = False
        iteration = 1
        rewards = []
        while (iteration <= max_iterations) and (not done):
            # get q values
            q = []
            for j in range(3):
                temp_q = bias
                for k, v in state.items():
                    temp_q += (weights[k][j] * v)
                q.append(temp_q)

            # get exploit action based on q values
            max_q_val = q[0]
            exploit_action = 0
            for k in range(1, 3):
                if q[k] > max_q_val:
                    max_q_val = q[k]
                    exploit_action = k

            # get actual action based on epsilon value and q value based on the action and current state
            action = np.random.choice([exploit_action, 0, 1, 2], 1,
                                      p=[1 - epsilon, epsilon / 3, epsilon / 3, epsilon / 3])[0]
            q_val = q[action]
            old_state = state

            # perform next step
            state, reward, done = mc.step(action)
            rewards.append(reward)

            # fetch max next state q value
            q = []
            for j in range(3):
                temp_q = bias
                for k, v in state.items():
                    temp_q += (weights[k][j] * v)
                q.append(temp_q)
            max_next_q_val = max(q)

            # update the weights and bias based on function approx rule
            first_term = q_val - (reward + (gamma * max_next_q_val))
            for k, v in old_state.items():
                weights[k][action] -= learning_rate * first_term * v
            bias -= learning_rate * first_term

            iteration += 1
        return_rewards.append(sum(rewards))


        # if i >= 24:
        #     avg_rewards.append(sum(return_rewards[i - 24: i + 1]) / 25.0)
        # else:
        #     avg_rewards.append(sum(return_rewards) / float(len(return_rewards)))


    # x = range(len(return_rewards))
    #
    # plt.plot(x, return_rewards, label='Return per episode')
    # plt.plot(x, avg_rewards, label='Rolling Mean')

    # plt.xlabel('Number of episodes')
    # plt.ylabel('Return')
    # plt.title("Return vs Number of episodes: Raw features")
    # # plt.axis([0, 2.25, 1, 100])
    # plt.legend()
    # plt.show()

    # plt.xlabel('Number of episodes')
    # plt.ylabel('Return')
    # plt.title("Return vs Number of episodes: Tile features")
    # # plt.axis([0, 2.25, 1, 100])
    # plt.legend()
    # plt.show()

    with open(return_out_file, "w") as f:
        for return_reward in return_rewards:
            f.write(str(return_reward))
            f.write("\n")

    with open(weight_out_file, "w") as f:
        f.write(str(bias))
        f.write("\n")
        for state in weights:
            for action in state:
                f.write(str(action))
                f.write("\n")


if __name__ == "__main__":
    main(sys.argv)
