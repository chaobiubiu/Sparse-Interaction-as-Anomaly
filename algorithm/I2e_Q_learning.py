import numpy as np
import torch as th
from model.vae_network import VAE


class I2e_Q_learning:
    def __init__(self, args, state_num, action_num):
        super(I2e_Q_learning, self).__init__()
        self.args = args
        self.Q_value = np.zeros([state_num, action_num])
        self.lr = args.lr

        self.action_num = action_num
        state_dim = args.map_size * args.map_size
        action_dim = 5

        state_embed_dim = args.state_embedding_size
        action_embed_dim = args.action_embedding_size
        latent_dim = args.latent_dim
        self.vae = VAE(state_dim, action_dim, state_embed_dim, action_embed_dim, latent_dim, args)
        if args.use_cuda:
            self.vae.cuda(args.device)
        self.vae_optimizer = th.optim.Adam(self.vae.parameters(), lr=args.lr_vae)

        self.learn_step = 0
        self.hidden_state = None

    def init_hidden(self, batch_size):
        self.hidden_state = self.vae.init_hidden(batch_size)

    def choose_action(self, state, epsilon):
        action_value = self.Q_value[state]
        max_action = np.where(action_value == np.max(action_value))
        action = max_action[0][np.random.randint(0, len(max_action))]       # When multiple equal values exist, randomly select one
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, state, action, reward, next_state):
        self.Q_value[state, action] += self.lr * (reward + 0.99 * np.max(self.Q_value[next_state]) - self.Q_value[state, action])
        self.learn_step += 1

    def vae_predict(self, curr_state, curr_action):
        curr_state = th.from_numpy(np.array(curr_state)).unsqueeze(dim=0).unsqueeze(dim=0).long()       # (1, 1, 1)
        curr_action = th.from_numpy(np.array(curr_action)).unsqueeze(dim=0).unsqueeze(dim=0).long()     # (1, 1, 1)
        latents, mu, log_vars, self.hidden_state = self.vae.encoder_forward(curr_state, curr_action, hidden_state=self.hidden_state)
        rec_next_states, rec_rewards = self.vae.decoder(latents, curr_state, curr_action)
        return rec_next_states, rec_rewards

    def vae_learn(self, replay_buffer, agent_index, logger, iter, curr_steps):
        if replay_buffer.__len__() > self.args.batch_size:
            episode_data = replay_buffer.sample(self.args.batch_size)
            # Initialize the hidden states as (1, batch_size, gru_hidden_size)
            self.init_hidden(self.args.batch_size)
            # shape=(bs, seq_length, ...)
            s_trajectory = th.from_numpy(np.array(episode_data['s'])).long()
            a_trajectory = th.from_numpy(np.array(episode_data['a'])).long()
            r_trajectory = th.from_numpy(np.array(episode_data['r'])).float()
            s_n_trajectory = th.from_numpy(np.array(episode_data['s_'])).long()
            mask_trajectory = th.from_numpy(np.array(episode_data['mask'])).float()

            if self.args.use_cuda:
                s_trajectory = s_trajectory.to(self.args.device)
                a_trajectory = a_trajectory.to(self.args.device)
                r_trajectory = r_trajectory.to(self.args.device)
                s_n_trajectory = s_n_trajectory.to(self.args.device)
                mask_trajectory = mask_trajectory.to(self.args.device)

            latents, mu, log_vars, self.hidden_state = self.vae.encoder_forward(s_trajectory, a_trajectory, hidden_state=self.hidden_state)
            rec_next_states_loss, rec_rewards_loss, kl_loss = self.vae.calculate_loss(latents, mu, log_vars, s_trajectory, a_trajectory, s_n_trajectory, r_trajectory, mask_trajectory)
            vae_loss = rec_next_states_loss + rec_rewards_loss + kl_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            logger.add_scalar('iter_{}_agent_{}_pred_next_state_loss'.format(iter, agent_index), rec_next_states_loss, curr_steps)
            logger.add_scalar('iter_{}_agent_{}_pred_reward_loss'.format(iter, agent_index), rec_rewards_loss, curr_steps)
            logger.add_scalar('iter_{}_agent_{}_kl_loss'.format(iter, agent_index), kl_loss, curr_steps)

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q

    def save_models(self, iter, agent_index, path):
        np.save(path + "/iter_{}_agent_{}_Q_value.npy".format(iter, agent_index), self.Q_value)
        th.save(self.vae.state_dict(), path + "/iter_{}_agent_{}_vae.th".format(iter, agent_index))

    def load_vae_models(self, path, seed_number, agent_index):
        self.vae.load_state_dict(th.load("{}/iter_{}_agent_{}_vae.th".format(path, seed_number, agent_index), map_location=lambda storage, loc: storage))