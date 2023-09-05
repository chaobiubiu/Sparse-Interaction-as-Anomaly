import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(VFAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_dq = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.fc_iq = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # q = self.fc2(h)
        dq = self.fc_dq(h)
        iq = self.fc_iq(h)
        return dq, iq, h
