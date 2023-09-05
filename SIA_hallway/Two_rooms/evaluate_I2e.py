import os
import sys
sys.path.append("..")
import argparse
import random
import datetime
import torch as th
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.I2e_Q_learning import I2e_Q_learning
from algorithm.I2e_replay_buffer import I2e_Buffer
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser("Experiments of I2e")
# Core training parameters
parser.add_argument("--buffer_size", type=int, default=int(500), help="number of transitions can be stored in buffer")
parser.add_argument("--batch_size", type=int, default=16, help="number of sampled episodes to optimize")
parser.add_argument("--lr", type=float, default=0.7, help="learning rate of q")

parser.add_argument("--gru_hidden_size", type=int, default=64)

parser.add_argument("--epsilon_start", type=float, default=1.0)
parser.add_argument("--epsilon_finish", type=float, default=0.05)
parser.add_argument("--epsilon_anneal_time", type=int, default=10000)

parser.add_argument("--lr_vae", type=float, default=1e-3, help="learning rate of vae")
parser.add_argument("--latent_dim", type=int, default=16)
parser.add_argument("--state_embedding_size", type=int, default=10)
parser.add_argument("--action_embedding_size", type=int, default=10)
parser.add_argument("--bonus_weight", type=float, default=0.1)

args = parser.parse_args()


if __name__ == '__main__':
    max_episode = 1
    max_iteration = 1
    args.max_steps = 50         # total_steps = 500 * 50=25000 steps
    map_size = 7
    args.map_size = map_size

    evaluate = True

    args.use_cuda = True
    args.device = "cuda:2" if args.use_cuda else "cpu"

    delta_epsilon = (args.epsilon_start - args.epsilon_finish) / args.epsilon_anneal_time

    steps_mean = np.zeros([1, max_iteration, max_episode])
    rewards_agent = np.zeros([1, max_iteration, max_episode])
    for num_agent in [2]:
        print("The number of agents", num_agent)
        env = EnvGoObstacle(map_size, num_agent)

        log_path = 'runs/initial_two_rooms/' + 'n_agents_{}'.format(num_agent) + '/' + \
                   '0822night_I2e_lr_{}__epsilon_anneal_{}_batch_size_{}_bonus_weight_{}_{}'.format(args.lr, args.epsilon_anneal_time, args.batch_size, args.bonus_weight,
                                                                                          datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        logger = None
        if not evaluate:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            logger = SummaryWriter(log_path)

        I2e_agents = [I2e_Q_learning(args, map_size * map_size, 5) for _ in range(num_agent)]
        Repaly_buffer = [I2e_Buffer(args) for _ in range(num_agent)]
        # Record visitation count of each state in the map
        visitation_matrix = np.zeros([max_iteration, max_episode, map_size, map_size])

        # policy_differ_num = np.zeros([max_episode, map_size, map_size])
        # z_num = np.zeros([max_episode, map_size, map_size])
        for iter in range(max_iteration):

            seed = np.random.randint(0, 999999)
            random.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)

            seed_number = np.random.randint(0, 10)
            print("Seed number is", seed_number)

            # Initialize Q with single-agent optimal Q
            for index in range(num_agent):
                if index == 0:
                    Q = np.load("I2e_results/iter_{}_agent_0_Q_value.npy".format(seed_number))
                elif index == 1:
                    Q = np.load("I2e_results/iter_{}_agent_1_Q_value.npy".format(seed_number))
                else:
                    raise Exception("Currently consider only two agent setting.")
                I2e_agents[index].set_Q_value(Q)
                I2e_agents[index].load_vae_models("I2e_results", seed_number, index)

            all_steps = 0
            for i in tqdm(range(max_episode)):
                state = env.reset()     # [[x, y] for _ in range(num_agent)]
                count = 0
                done = False
                rewards = np.zeros(num_agent)

                record_rewards = [[] for _ in range(num_agent)]
                record_bonus_rewards = [[] for _ in range(num_agent)]
                record_state_diff = [[] for _ in range(num_agent)]
                record_rew_diff = [[] for _ in range(num_agent)]

                state_stack = [[] for _ in range(num_agent)]
                action_stack = [[] for _ in range(num_agent)]
                reward_stack = [[] for _ in range(num_agent)]
                next_state_stack = [[] for _ in range(num_agent)]
                mask_stack = [[] for _ in range(num_agent)]
                # Initialize the hidden states
                for index in range(num_agent):
                    I2e_agents[index].init_hidden(batch_size=1)

                while not done:
                    action_list = []
                    # if i > max_episode-3:
                    #     env.render()
                    # epsilon = np.min([0.95, args.min_epsilon + (0.95 - args.min_epsilon) * (i * args.max_steps + count) / (max_episode * args.max_steps / 3)])
                    # epsilon = max(args.epsilon_finish, args.epsilon_start - delta_epsilon * (i * args.max_steps + count))
                    epsilon = 0.0
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]

                        visitation_matrix[iter][i][state[index][0]][state[index][1]] += 1

                        action = I2e_agents[index].choose_action(s, epsilon)
                        # z_num[i][state[index][0]][state[index][1]] = (z_num[i][state[index][0]][state[index][1]] * (visited_num[i][state[index][0]][state[index][1]] - 1) + z_norm) / visited_num[i][state[index][0]][state[index][1]]
                        action_list.append(action)
                        # if flag == 1:
                        #     policy_differ_num[i][state[index][0]][state[index][1]] += 1

                    reward, done, next_state = env.step(action_list)

                    print("Step", count, state, action_list, reward, done, next_state)

                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0] * map_size + next_state[index][1]
                        state_stack[index].append(s)
                        action_stack[index].append(a)
                        reward_stack[index].append(r)
                        next_state_stack[index].append(s_)
                        rewards[index] += r
                        record_rewards[index].append(r)
                        mask_stack[index].append(1)
                        # Here we design intrinsic rewards to encourage exploration towards interactive areas
                        with th.no_grad():
                            rec_next_states, rec_rewards = I2e_agents[index].vae_predict(s, a)
                            # If reconstructed elements significantly differ from their counterpart, then interaction occurs
                            label_s_ = th.from_numpy(np.array(s_)).unsqueeze(dim=0).unsqueeze(dim=0)
                            label_s_ = F.one_hot(label_s_, num_classes=map_size * map_size).float()
                            label_r_ = th.from_numpy(np.array(r)).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).float()

                            if args.use_cuda:
                                label_s_ = label_s_.to(args.device)
                                label_r_ = label_r_.to(args.device)
                            state_diff = th.mean((rec_next_states - label_s_) ** 2, dim=-1, keepdim=True)    # (1, 1, 1)
                            # state_diff = ((rec_next_states - s_) ** 2).mean()
                            reward_diff = th.abs(label_r_ - rec_rewards)        # (1, 1, 1)
                            explore_bonus = (state_diff + reward_diff) * args.bonus_weight
                            explore_bonus = explore_bonus[0][0][0].cpu().numpy()
                            record_bonus_rewards[index].append(explore_bonus)
                            record_state_diff[index].append(state_diff[0][0][0].cpu().numpy())
                            record_rew_diff[index].append(reward_diff[0][0][0].cpu().numpy())
                            shaped_r = r + explore_bonus
                            print("explore_bonus", index, explore_bonus)
                        # Train each agent's policy
                        if not evaluate:
                            I2e_agents[index].learn(s, a, shaped_r, s_)
                            print("Train occurs.")

                    state = next_state
                    count += 1

                    print("====================================")

                    if count >= args.max_steps - 1:
                        done = True

                if count <= args.max_steps - 1:
                    remaining_steps = args.max_steps - count
                    for remaining_step in range(remaining_steps):
                        for index in range(num_agent):
                            state_stack[index].append(0)
                            action_stack[index].append(0)
                            reward_stack[index].append(0)
                            next_state_stack[index].append(0)
                            mask_stack[index].append(0)

                # for index in range(num_agent):
                #     Repaly_buffer[index].add(np.array(state_stack[index]), np.array(action_stack[index]),
                #                              np.array(reward_stack[index]), np.array(next_state_stack[index]),
                #                              np.array(mask_stack[index]))
                # if i % 1 == 0:
                #     for index in range(num_agent):
                #         I2e_agents[index].vae_learn(Repaly_buffer[index], index, logger, iter, i)

                # bonus_rewards store the shaped intrinsic rewards of all agents
                if logger is not None:
                    for index in range(num_agent):
                        logger.add_scalar('iter_{}_agent_{}_explore_bonus_mean'.format(iter, index), np.mean(record_bonus_rewards[index]), i)
                        logger.add_scalar('iter_{}_agent_{}_reward_mean'.format(iter, index), np.mean(record_rewards[index]), i)
                        logger.add_scalar('iter_{}_agent_{}_state_diff_mean'.format(iter, index),
                                          np.mean(record_state_diff[index]), i)
                        logger.add_scalar('iter_{}_agent_{}_rew_diff_mean'.format(iter, index),
                                          np.mean(record_rew_diff[index]), i)

                rewards_agent[0][iter][i] = np.mean(rewards)        # total reward calculation
                print("iter_{}_episode_{}_episodic_reward".format(iter, i), count, rewards[0], rewards[1])
                if logger is not None:
                    logger.add_scalar('iter_{}_episodic_reward'.format(iter), np.mean(rewards), i)
                    logger.add_scalar('iter_{}_epsilon'.format(iter), epsilon, i)

                    # Per episode, record the episodic step
                    logger.add_scalar('iter_{}_step'.format(iter), count, i)

                # print(i)
                all_steps += count
                # steps_mean[num_agent - 5, iter] = all_steps

            if not evaluate:
                for index in range(num_agent):
                    I2e_agents[index].save_models(iter, index, "I2e_results")

    if logger is not None:
        logger.close()
    # np.save("WToE_rewards.npy", rewards_agent)
    # np.save("WToE_steps.npy", steps_mean)


    # np.save("WToE_rewards_" + str(num_agent) + ".npy", Rewards)
    # visited_num[:,0,0] = 0
    # visited_num[:,map_size-1,map_size-1] = 0
    # policy_differ_num[:,0,0] = 0
    # policy_differ_num[:,map_size-1,map_size-1] = 0
    # z_num[:,0,0] = 0
    # z_num[:,map_size-1,map_size-1] = 0
    # # visited_num = visited_num/max_iteration
    # # policy_differ_num = policy_differ_num/max_iteration
    # z_num = z_num/max_iteration

    # Save the visitation number
    # np.save("I2e_results/visitation_matrix.npy", visitation_matrix)


    # # np.save("differ.npy", policy_differ_num)
    # np.save("z.npy", z_num)
    # sns.set()
    #
    # # plt.figure(1)
    # # ax = sns.heatmap(visited_num, fmt='d', linewidths=.5, cmap='YlGnBu')
    # # plt.savefig("heat_map_visited.pdf")
    # #
    # # plt.figure(2)
    # # ax2 = sns.heatmap(policy_differ_num, fmt='d', linewidths=.5, cmap='YlGnBu')
    # # plt.savefig("heat_map_policy.pdf")
    #
    # plt.figure(1)
    # ax2 = sns.heatmap(z_num, fmt='d', linewidths=.5, cmap='YlGnBu')
    # plt.savefig("z_num.pdf")