import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super(Encoder, self).__init__()
        self.args = args
        self.fc1_obs = nn.Sequential(nn.Linear(obs_dim, args.obs_embedding_dim), nn.ReLU())
        self.fc1_act = nn.Sequential(nn.Linear(act_dim, args.act_embedding_dim), nn.ReLU())
        self.rnn = nn.GRU((args.obs_embedding_dim + args.act_embedding_dim), args.hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(args.hidden_size, args.latent_dim)
        self.fc_var = nn.Linear(args.hidden_size, args.latent_dim)

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
        return self.fc_mu.weight.new(1, self.args.hidden_size).zero_()

    def forward(self, observations, actions):
        # observations/actions.shape=(bs, max_seq_length-1, n_agents, dim)
        bs, seq_length, n_agents, _ = observations.size()
        obs_embeddings = self.fc1_obs(observations)
        act_embeddings = self.fc1_act(actions)
        embeddings = th.cat([obs_embeddings, act_embeddings], dim=-1)       # (bs, max_seq_length-1, n_agents, rnn_hidden_dim)
        embeddings = embeddings.permute(0, 2, 1, 3).reshape(bs*n_agents, seq_length, -1)
        h0 = self.init_hidden().expand(bs*n_agents, -1).unsqueeze(dim=0).contiguous()       # (1, bs*n_agents, hidden_size)
        output, hn = self.rnn(embeddings, h0)   # output.shape=(bs*n_agents, seq_length, hidden_size)
        output = output.reshape(bs, n_agents, seq_length, -1).permute(0, 2, 1, 3)   # (bs, seq_length, n_agents, hidden_size)
        mu = self.fc_mu(output)     # (bs, seq_length, n_agents, latent_dim)
        log_var = self.fc_var(output)       # The same shape as mu
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, act_dim, args):
        super(Decoder, self).__init__()
        self.fc1_obs = nn.Sequential(nn.Linear(obs_dim, args.obs_embedding_dim), nn.ReLU())     # Embedding Layer
        self.fc1_act = nn.Sequential(nn.Linear(act_dim, args.act_embedding_dim), nn.ReLU())
        concat_dims = args.obs_embedding_dim + args.act_embedding_dim
        self.next_obs_pred = nn.Sequential(nn.Linear(concat_dims + latent_dim, args.hidden_size), nn.ReLU(),
                                           nn.Linear(args.hidden_size, obs_dim))
        self.reward_pred = nn.Sequential(nn.Linear(concat_dims + latent_dim, args.hidden_size), nn.ReLU(),
                                         nn.Linear(args.hidden_size, 1))

    def forward(self, latents, curr_obs, curr_act):
        # latents.shape=(bs, seq_length, n_agents, latent_dim)
        obs_embeddings = self.fc1_obs(curr_obs)
        act_embeddings = self.fc1_act(curr_act)
        concat_inps = th.cat([obs_embeddings, act_embeddings, latents], dim=-1)
        pred_next_obs = self.next_obs_pred(concat_inps)       # (bs, seq_length, n_agents, obs_dim)
        pred_reward = self.reward_pred(concat_inps)       # (bs, seq_length, n_agents, 1)
        return pred_next_obs, pred_reward


class VAE(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super(VAE, self).__init__()
        self.encoder = Encoder(obs_dim, act_dim, args)
        self.decoder = Decoder(args.latent_dim, obs_dim, act_dim, args)

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def encoder_forward(self, curr_obs, curr_act):
        mu, log_var = self.encoder(curr_obs, curr_act)
        latent = self.reparameterize(mu, log_var)
        return latent, mu, log_var

    def calculate_loss(self, latent, mu, log_var, curr_obs, curr_act, next_obs, rewards):
        # curr_obs/curr_act/next_obs/rewards.shape = (bs, max_seq_length-1, n_agents, -1)
        pred_next_obs, pred_rewards = self.decoder(latent, curr_obs, curr_act)
        # pred_next_obs / pred_rewards.shape=(bs, seq_length, n_agents, obs_dim / 1)
        rec_obs_loss = F.mse_loss(pred_next_obs, next_obs, reduction='none').mean(dim=-1, keepdim=True)
        rec_reward_loss = F.mse_loss(pred_rewards, rewards, reduction='none').mean(dim=-1, keepdim=True)
        kl_loss = - 0.5 * (1 + log_var - mu ** 2 - log_var.exp()).mean(dim=-1, keepdim=True)
        return rec_obs_loss, rec_reward_loss, kl_loss