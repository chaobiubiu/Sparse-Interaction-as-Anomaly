import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, state_embed_dim, action_embed_dim, args):
        super(Encoder, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_state = nn.Sequential(nn.Linear(state_dim, state_embed_dim), nn.ReLU())
        self.fc1_action = nn.Sequential(nn.Linear(action_dim, action_embed_dim), nn.ReLU())
        total_input_dim = state_embed_dim + action_embed_dim
        self.rnn = nn.GRU(total_input_dim, args.gru_hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(args.gru_hidden_size, latent_dim)
        self.fc_var = nn.Linear(args.gru_hidden_size, latent_dim)

        # GRU-related settings
        # Args:
        #         input_size: The number of expected features in the input `x`
        #         hidden_size: The number of features in the hidden state `h`
        #         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        #             would mean stacking two GRUs together to form a `stacked GRU`,
        #             with the second GRU taking in outputs of the first GRU and
        #             computing the final results. Default: 1
        #         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        #             Default: ``True``
        #         batch_first: If ``True``, then the input and output tensors are provided
        #             as (batch, seq, feature). Default: ``False``
        #         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        #             GRU layer except the last layer, with dropout probability equal to
        #             :attr:`dropout`. Default: 0
        #         bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``
        # Inputs: input, h_0
        #         - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
        #           of the input sequence. The input can also be a packed variable length
        #           sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
        #           for details.
        #         - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #           containing the initial hidden state for each element in the batch.
        #           Defaults to zero if not provided. If the RNN is bidirectional,
        #           num_directions should be 2, else it should be 1.

        # Outputs: output, h_n
        #         - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
        #           containing the output features h_t from the last layer of the GRU,
        #           for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
        #           given as the input, the output will also be a packed sequence.
        #           For the unpacked case, the directions can be separated
        #           using ``output.view(seq_len, batch, num_directions, hidden_size)``,
        #           with forward and backward being direction `0` and `1` respectively.
        #
        #           Similarly, the directions can be separated in the packed case.
        #         - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #           containing the hidden state for `t = seq_len`
        #
        #           Like *output*, the layers can be separated using
        #           ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

        # Examples::
        #
        #         >>> rnn = nn.GRU(10, 20, 2)
        #         >>> input = torch.randn(5, 3, 10)
        #         >>> h0 = torch.randn(2, 3, 20)
        #         >>> output, hn = rnn(input, h0)

    def init_hidden(self):
        # Make hidden states on same device as model
        return self.fc_mu.weight.new(1, self.args.gru_hidden_size).zero_()

    def forward(self, curr_state, curr_action, hidden_state):
        # When conduct Q-table update, we feed states and actions step by step.
        if len(curr_state.size()) == 2:
            bs, seq_length = curr_state.size()
        else:
            bs, seq_length, _ = curr_state.size()
            curr_state = curr_state.squeeze(dim=-1)
            curr_action = curr_action.squeeze(dim=-1)
        state_inps = F.one_hot(curr_state, num_classes=self.state_dim).float()
        act_inps = F.one_hot(curr_action, num_classes=self.action_dim).float()
        if self.args.use_cuda:
            state_inps = state_inps.to(self.args.device)
            act_inps = act_inps.to(self.args.device)
        state_embedding = self.fc1_state(state_inps)     # (bs, seq_length, state_embed_dim)
        act_embedding = self.fc1_action(act_inps)       # (bs, seq_length, act_embed_dim)
        concat_inps = th.cat([state_embedding, act_embedding], dim=-1)      # (bs, seq_length, state_embed_dim + act_embed_dim)
        output, hidden_state = self.rnn(concat_inps, hidden_state)      # output.shape=(bs, seq_length, gru_hidden_size)
        mu = self.fc_mu(output)     # (bs, seq_length, latent_dim)
        log_var = self.fc_var(output)       # (bs, seq_length, latent_dim)
        return mu, log_var, hidden_state


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_dim, action_dim, state_embed_dim, action_embed_dim, args):
        super(Decoder, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1_state = nn.Sequential(nn.Linear(state_dim, state_embed_dim), nn.ReLU())
        self.fc1_action = nn.Sequential(nn.Linear(action_dim, action_embed_dim), nn.ReLU())

        total_input_dim = state_embed_dim + action_embed_dim + latent_dim
        self.next_state_pred = nn.Sequential(nn.Linear(total_input_dim, args.gru_hidden_size), nn.ReLU(),
                                             nn.Linear(args.gru_hidden_size, state_dim))
        self.reward_pred = nn.Sequential(nn.Linear(total_input_dim, args.gru_hidden_size), nn.ReLU(),
                                         nn.Linear(args.gru_hidden_size, 1))

    def forward(self, latents, curr_state, curr_action):
        # When conduct Q-table update, we feed states and actions step by step.
        if len(curr_state.size()) == 2:
            bs, seq_length = curr_state.size()
        else:
            bs, seq_length, _ = curr_state.size()
            curr_state = curr_state.squeeze(dim=-1)
            curr_action = curr_action.squeeze(dim=-1)
        state_inps = F.one_hot(curr_state, num_classes=self.state_dim).float()
        act_inps = F.one_hot(curr_action, num_classes=self.action_dim).float()
        if self.args.use_cuda:
            state_inps = state_inps.to(self.args.device)
            act_inps = act_inps.to(self.args.device)
        state_embedding = self.fc1_state(state_inps)  # (bs, seq_length, state_embed_dim)
        act_embedding = self.fc1_action(act_inps)  # (bs, seq_length, act_embed_dim)
        concat_inps = th.cat([state_embedding, act_embedding, latents], dim=-1)  # (bs, seq_length, state_embed_dim + act_embed_dim + latent_dim)
        pred_next_state = self.next_state_pred(concat_inps)     # (bs, seq_length, state_dim)
        pred_reward = self.reward_pred(concat_inps)     # (bs, seq_length, 1)
        return pred_next_state, pred_reward


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, state_embed_dim, action_embed_dim, latent_dim, args):
        super(VAE, self).__init__()
        self.state_dim = state_dim
        self.encoder = Encoder(state_dim, action_dim, latent_dim, state_embed_dim, action_embed_dim, args)
        self.decoder = Decoder(latent_dim, state_dim, action_dim, state_embed_dim, action_embed_dim, args)

    def init_hidden(self, bs):
        hidden_states = self.encoder.init_hidden().expand(bs, -1).unsqueeze(dim=0).contiguous()      # (1, bs, gru_hidden_size)
        return hidden_states

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def encoder_forward(self, curr_state, curr_action, hidden_state):
        mu, log_var, hidden_state = self.encoder(curr_state, curr_action, hidden_state)
        latent = self.reparameterize(mu, log_var)
        return latent, mu, log_var, hidden_state      # (bs, seq_length, latent_dim)

    def calculate_loss(self, latents, mu, log_var, curr_states, curr_acts, next_states, rewards, mask):
        # input.shape = (bs, seq_length, latent_dim/state_dim/n_actions/state_dim/1)
        # latent, mu, log_var = self.encoder_forward(segment)
        rec_next_states, rec_rewards = self.decoder(latents, curr_states, curr_acts)
        label_next_states = F.one_hot(next_states.squeeze(dim=-1), num_classes=self.state_dim).float()
        rec_next_states_loss = F.mse_loss(rec_next_states, label_next_states, reduction='none').mean(dim=-1, keepdim=True)
        # rec_next_states_loss = F.mse_loss(rec_next_states, next_states)
        rec_rewards_loss = F.mse_loss(rec_rewards, rewards, reduction='none').mean(dim=-1, keepdim=True)
        # kl_loss = th.mean(- 0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kl_loss = - 0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1, keepdim=True)
        masked_rec_next_states_loss = (rec_next_states_loss * mask).sum() / mask.sum()
        masked_rec_rewards_loss = (rec_rewards_loss * mask).sum() / mask.sum()
        masked_kl_loss = (kl_loss * mask).sum() / mask.sum()
        return masked_rec_next_states_loss, masked_rec_rewards_loss, masked_kl_loss