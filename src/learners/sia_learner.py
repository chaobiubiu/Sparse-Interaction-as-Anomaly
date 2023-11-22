import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
from src.modules.auxiliary_nets.vae import VAE
import torch as th
from torch.optim import RMSprop
from torch.distributions import Categorical


class SIALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.initial_aux_weight = args.aux_weight
        self.aux_weight = args.aux_weight
        self.bonus_weight = args.bonus_weight
        self.use_weight_decay = args.use_weight_decay
        self.anneal_rate = args.anneal_rate
        self.t_max = args.t_max

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.vae = VAE(args.obs_shape, args.n_actions, args)
        self.params += list(self.vae.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # =======================  Begin VAE Calculation  =========================
        curr_obs = batch["obs"][:, :-1]  # (bs, max_seq_length-1, n_agents, obs_shape)
        next_obs = batch["obs"][:, 1:]  # (bs, max_seq_length-1, n_agents, obs_shape)
        actions_onehot = batch["actions_onehot"][:, :-1]  # (bs, max_seq_length-1, n_agents, n_actions)
        # rewards.shape=(bs, max_seq_length-1, 1)
        curr_rewards = rewards.unsqueeze(dim=2).expand(-1, -1, self.args.n_agents, -1)      # (bs, max_seq_length-1, n_agents, 1)

        latents, mu, log_vars = self.vae.encoder_forward(curr_obs, actions_onehot)
        rec_next_obs_loss, rec_reward_loss, kl_loss = self.vae.calculate_loss(latents, mu, log_vars, curr_obs,
                                                                              actions_onehot, next_obs, curr_rewards)

        with th.no_grad():
            state_diff = th.mean(rec_next_obs_loss, dim=2)      # (bs, max_seq_length-1, 1)
            reward_diff = th.mean(rec_reward_loss, dim=2)       # (bs, max_seq_length-1, 1)
            explore_bonus = state_diff + self.bonus_weight * reward_diff        # (bs, max_seq_length-1, 1)
            explore_bonus = explore_bonus * self.aux_weight

        if self.use_weight_decay:
            self.aux_weight = max((1 - (self.anneal_rate * t_env / self.t_max)), 0) * self.initial_aux_weight

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # mac_out_dist = Categorical(logits=mac_out.clone().detach()).entropy()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        shaped_rewards = rewards + explore_bonus

        # Calculate 1-step Q-Learning targets
        targets = shaped_rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # print((targets == rewards).all())

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # vae loss
        # rec_next_obs_loss, rec_reward_loss, kl_loss.shape=(bs, max_seq_length-1, n_agents, 1)
        curr_mask = mask.unsqueeze(dim=2).expand(-1, -1, self.args.n_agents, -1)
        masked_rec_next_obs_loss = (rec_next_obs_loss * curr_mask).sum() / curr_mask.sum()
        masked_rec_reward_loss = (rec_reward_loss * curr_mask).sum() / curr_mask.sum()
        masked_kl_loss = (kl_loss * curr_mask).sum() / curr_mask.sum()

        vae_loss = masked_rec_next_obs_loss + masked_rec_reward_loss + masked_kl_loss

        total_loss = loss + vae_loss

        # Optimise
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("masked_rec_next_obs_loss", masked_rec_next_obs_loss.item(), t_env)
            self.logger.log_stat("masked_rec_reward_loss", masked_rec_reward_loss.item(), t_env)
            self.logger.log_stat("masked_kl_loss", masked_kl_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)

            self.logger.log_stat("state_diff", th.mean(state_diff).item(), t_env)
            self.logger.log_stat("reward_diff", th.mean(reward_diff).item(), t_env)
            self.logger.log_stat("explore_bonus", th.mean(explore_bonus).item(), t_env)
            self.logger.log_stat("curr_rewards", th.mean(curr_rewards).item(), t_env)

            self.logger.log_stat("aux_weight", self.aux_weight, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.vae.to(self.args.device)
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.vae.state_dict(), "{}/vae.th".format(path))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.vae.load_state_dict(th.load("{}/vae.th".format(path), map_location=lambda storage, loc: storage))
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def show_matrix_info(self, batch, t_env):
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # agent_outs = self.mac.forward(batch, t=t, show_h=bool(1-t))
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time, threads, steps, agents, actions
        actions_dim = mac_out.shape[3]
        print("Episode %i, The learned matrix payoff is:" % t_env)
        payoff = ""
        for ai in range(actions_dim):
            for aj in range(actions_dim):
                actions = th.tensor([[ai, aj]]).to(**dict(dtype=th.int64, device=mac_out.device))
                actions = actions.unsqueeze(0).unsqueeze(-1).repeat(mac_out.shape[0], mac_out.shape[1]-1, 1, 1)
                # print(actions.shape, actions)
                chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
                if self.mixer is not None:
                    # if ai == 0 and aj == 0:
                    #     mixer_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], True)
                    # else:
                    mixer_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                else:
                    mixer_qvals = th.zeros((1, 1, 1))
                sp = "{0:.4}".format(str(chosen_action_qvals[0, 0, 0].item())) + "||" \
                     + "{0:.4}".format(str(chosen_action_qvals[0, 0, 1].item())) \
                     + "||" + "{0:.4}".format(str(mixer_qvals[0, 0, 0].item())) + "     "
                payoff += sp
                # print(ai, aj, chosen_action_qvals[0, 0, 0].item(),
                # chosen_action_qvals[0, 0, 1].item(), mixer_qvals[0, 0, 0].item())
            payoff += "\n"
        print(payoff)
        # max_actions = mac_out.max(dim=3)[1]
        max_actions = batch["actions"][:, :-1, :, 0]
        print("Max actions is:", max_actions[0, 0, 0].item(), max_actions[0, 0, 1].item(),
              "  ||   Reward is", batch["reward"][0, 0, 0].item())
        # chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        # print(mac_out.shape)
        # print(mac_out)
        # print(batch["actions"][:, :-1])
        # print(batch["actions"][:, :-1].shape)
        # exit()
        # print(self.mixer.state_dict())

    def show_mmdp_info(self, batch, t_env):
        pass
