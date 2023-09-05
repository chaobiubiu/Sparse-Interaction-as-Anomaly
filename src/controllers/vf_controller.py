from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicVFMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.kao = args.kao
        self.kao_end = args.kao_end
        self.kao_gap = self.kao - self.kao_end
        self.kao_anneal_time = args.kao_anneal_time
        self.t_env = 0

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_dq, agent_iq = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # print(agent_dq.shape, agent_iq.shape, agent_iq[0, 0, :], agent_dq[0, 0, :])
        if self.args.normalize:
            agent_dq = th.nn.functional.softmax(agent_dq, dim=2)
            agent_iq = th.nn.functional.softmax(agent_iq, dim=2)
        # print(agent_dq.shape, agent_iq.shape, agent_iq[0, 0, :], agent_dq[0, 0, :], test_mode)
        # if test_mode and not self.args.eval_use_iq:
        #     agent_outputs = agent_dq
        # else:
        #     agent_outputs = (1 - self.kao) * agent_dq + self.kao * agent_iq
        #     self.update_kao(t_env)
        if test_mode:
            if self.args.eval_use_iq:
                agent_outputs = (1 - self.kao_end) * agent_dq + self.kao_end * agent_iq
            else:
                agent_outputs = agent_dq
        else:
            agent_outputs = (1 - self.kao) * agent_dq + self.kao * agent_iq
            self.update_kao(t_env)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def update_kao(self, t_env):
        gap = self.kao_gap * (t_env - self.t_env) * 1. / self.kao_anneal_time
        self.kao -= gap
        if self.kao_gap > 0:
            self.kao = max(self.kao, self.kao_end)
        else:
            self.kao = min(self.kao, self.kao_end)
        self.t_env = t_env
        # print(self.t_env, self.kao)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        # avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.share_embedding:
            agent_dq, agent_iq, self.hidden_states[0] = self.agent[0](agent_inputs, self.hidden_states[0])
        else:
            agent_dq, self.hidden_states[0] = self.agent[0](agent_inputs, self.hidden_states[0])
            agent_iq, self.hidden_states[1] = self.agent[1](agent_inputs, self.hidden_states[1])

        # if self.agent_output_type == "pi_logits":
        #
        #     if getattr(self.args, "mask_before_softmax", True):
        #         # Make the logits for unavailable actions very negative to minimise their affect on the softmax
        #         reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
        #         agent_outs[reshaped_avail_actions == 0] = -1e10
        #
        #     agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        #     if not test_mode:
        #         # Epsilon floor
        #         epsilon_action_num = agent_outs.size(-1)
        #         if getattr(self.args, "mask_before_softmax", True):
        #             # With probability epsilon, we will pick an available action uniformly
        #             epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
        #
        #         agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
        #                        + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
        #
        #         if getattr(self.args, "mask_before_softmax", True):
        #             # Zero out the unavailable actions
        #             agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_dq.view(ep_batch.batch_size, self.n_agents, -1), agent_iq.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.share_embedding:
            self.hidden_states = [self.agent[0].init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)]  # bav
        else:
            self.hidden_states = [
                self.agent[0].init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1),
                self.agent[1].init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1),
            ]

    def parameters(self):
        pars = list(self.agent[0].parameters())
        if not self.args.share_embedding:
            pars += list(self.agent[1].parameters())
        return pars

    def load_state(self, other_mac):
        self.agent[0].load_state_dict(other_mac.agent[0].state_dict())
        if not self.args.share_embedding:
            self.agent[1].load_state_dict(other_mac.agent[1].state_dict())

    def cuda(self):
        self.agent[0].cuda()
        if not self.args.share_embedding:
            self.agent[1].cuda()

    def save_models(self, path):
        th.save(self.agent[0].state_dict(), "{}/agent0.th".format(path))
        if not self.args.share_embedding:
            th.save(self.agent[1].state_dict(), "{}/agent1.th".format(path))

    def load_models(self, path):
        self.agent[0].load_state_dict(th.load("{}/agent0.th".format(path), map_location=lambda storage, loc: storage))
        if not self.args.share_embedding:
            self.agent[1].load_state_dict(
                th.load("{}/agent1.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        if self.args.agent == "vf" and self.args.share_embedding:
            self.agent = [agent_REGISTRY[self.args.agent](input_shape, self.args)]
        elif self.args.agent == "rnn" and not self.args.share_embedding:
            self.agent = [agent_REGISTRY[self.args.agent](input_shape, self.args),
                          agent_REGISTRY[self.args.agent](input_shape, self.args)]
        else:
            raise NotImplementedError

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
