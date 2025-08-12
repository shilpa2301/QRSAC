import gtimer as gt
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from .risk import distortion_de
from .utils import LinearSchedule

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer



def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()

class QRSACTrainer(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            target_policy,
            zf1,
            zf2,
            target_zf1,
            target_zf2,
            fp=None,
            target_fp=None,
            discount=0.99,
            reward_scale=1.0,
            alpha=1.0,
            policy_lr=3e-4,
            zf_lr=3e-4,
            tau_type='iqn',
            fp_lr=1e-5,
            num_quantiles=32,
            risk_type='neutral',
            risk_param=0.,
            risk_param_final=None,
            risk_schedule_timesteps=1,
            optimizer_class=optim.Adam,
            soft_target_tau=5e-3,
            target_update_period=5,
            clip_norm=0.,
            use_automatic_entropy_tuning=False,
            target_entropy=None,
            n_step=7,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.target_policy = target_policy
        self.zf1 = zf1
        self.zf2 = zf2
        self.target_zf1 = target_zf1
        self.target_zf2 = target_zf2

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.tau_type = tau_type
        self.num_quantiles = num_quantiles

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.alpha = alpha

        self.zf_criterion = quantile_regression_loss

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.zf1_optimizer = optimizer_class(
            self.zf1.parameters(),
            lr=zf_lr,
        )
        self.zf2_optimizer = optimizer_class(
            self.zf2.parameters(),
            lr=zf_lr,
        )

        self.fp = fp
        self.target_fp = target_fp
        if self.tau_type == 'fqf':
            self.fp_optimizer = optimizer_class(
                self.fp.parameters(),
                lr=fp_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.clip_norm = clip_norm

        self.risk_type = risk_type
        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.n_step = n_step

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau


    def train_from_torch(self, batch, replay_buffer):
        indices = batch['indices']

        n_step_data=[]
        for index in indices:
            index = index.cpu().numpy()
            nstep_data_for_index = replay_buffer.nstep_batch(
            self.n_step,index)
            n_step_data.append(nstep_data_for_index)
        weights = torch.tensor([self.discount ** i for i in range(self.n_step)])
        weighted_sums = []
        for n_step_data_per_id in n_step_data:
            weighted_sum = torch.sum(n_step_data_per_id * weights.view(-1, 1), dim=0)
            weighted_sums.append(weighted_sum)
        sz = batch['rewards'].size(0)
        rewards = torch.stack(weighted_sums).view(sz,1).cuda() #discounted n step rewards

        # rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        gt.stamp('preback_start', unique=False)

        #get states S_{t+N}
        n_step_buffer_data = replay_buffer.get_n_step_data(self.n_step, indices)
        obs_at_nstep = n_step_buffer_data['observations'].cuda()
        actions_at_nstep = n_step_buffer_data['actions'].cuda()
        terminals_at_nstep = n_step_buffer_data['terminals'].cuda()
        next_obs_at_nstep = n_step_buffer_data['next_observations'].cuda()
        rewards_at_nstep = n_step_buffer_data['rewards'].cuda()

        """
        Update Alpha
        """

        new_actions_at_nstep, policy_mean_at_nstep, policy_log_std_at_nstep, log_pi_at_nstep, *_ = self.policy(
            obs_at_nstep,
            reparameterize=True,
            return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi_at_nstep + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha
        gt.stamp('preback_alpha', unique=False)
        """
        Update ZF
        """


        with torch.no_grad():

            new_next_actions_at_nstep, _, _, new_log_pi_at_nstep, *_ = self.target_policy(
                next_obs_at_nstep,
                reparameterize=True,
                return_log_prob=True,
            )

            risk_param = self.risk_schedule(self._n_train_steps_total)

            if self.risk_type == 'VaR':
                tau_ = ptu.ones_like(rewards) * risk_param
                q1_new_actions_at_nstep = self.zf1(obs_at_nstep, new_actions_at_nstep, tau_)
                q2_new_actions_at_nstep = self.zf2(obs_at_nstep, new_actions_at_nstep, tau_)
            else:
                with torch.no_grad():
                    new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs_at_nstep, new_actions_at_nstep, fp=self.fp)
                z1_new_actions_at_nstep = self.zf1(obs_at_nstep, new_actions_at_nstep, new_tau_hat)
                z2_new_actions_at_nstep = self.zf2(obs_at_nstep, new_actions_at_nstep, new_tau_hat)
                if self.risk_type in ['neutral', 'std']:
                    q1_new_actions_at_nstep = torch.sum(new_presum_tau * z1_new_actions_at_nstep, dim=1, keepdims=True)
                    q2_new_actions_at_nstep = torch.sum(new_presum_tau * z2_new_actions_at_nstep, dim=1, keepdims=True)
                    if self.risk_type == 'std':
                        q1_std_at_nstep = new_presum_tau * (z1_new_actions_at_nstep - q1_new_actions_at_nstep).pow(2)
                        q2_std_at_nstep = new_presum_tau * (z2_new_actions_at_nstep - q2_new_actions_at_nstep).pow(2)
                        q1_new_actions_at_nstep -= risk_param * q1_std_at_nstep.sum(dim=1, keepdims=True).sqrt()
                        q2_new_actions_at_nstep -= risk_param * q2_std_at_nstep.sum(dim=1, keepdims=True).sqrt()
                else:
                    with torch.no_grad():
                        risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                    q1_new_actions_at_nstep = torch.sum(risk_weights * new_presum_tau * z1_new_actions_at_nstep, dim=1, keepdims=True)
                    q2_new_actions_at_nstep = torch.sum(risk_weights * new_presum_tau * z2_new_actions_at_nstep, dim=1, keepdims=True)

            #ASSUMPTION 1 - since batches, take the largest no of min q vals as min k
            tensor1_lower = q1_new_actions_at_nstep < q2_new_actions_at_nstep  # True where tensor1 has lower values
            tensor2_lower = q2_new_actions_at_nstep < q1_new_actions_at_nstep  # True where tensor2 has lower values
            # Step 2: Count True values (lower values) in each row for both tensors
            tensor1_lower_counts = tensor1_lower.sum(dim=1)
            tensor2_lower_counts = tensor2_lower.sum(dim=1)
            # Step 3: Determine which tensor has more lower values in each row
            tensor1_lower_counts_sum = tensor1_lower_counts.sum(dim=0)
            tensor2_lower_counts_sum = tensor2_lower_counts.sum(dim=0)
            if (tensor1_lower_counts_sum > tensor2_lower_counts_sum):
                k_at_nstep=1
            else:
                k_at_nstep=2

            next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs_at_nstep, new_next_actions_at_nstep, fp=self.target_fp)

            #calculate only one value for m min , not both
            target_z1_values_at_nstep = self.target_zf1(next_obs_at_nstep, new_next_actions_at_nstep, next_tau_hat)
            target_z2_values_at_nstep = self.target_zf2(next_obs_at_nstep, new_next_actions_at_nstep, next_tau_hat)

            if k_at_nstep ==1:
                target_z_values_at_nstep = target_z1_values_at_nstep - alpha * new_log_pi_at_nstep
            else:
                target_z_values_at_nstep = target_z2_values_at_nstep - alpha * new_log_pi_at_nstep
            #in paper target z not multiplied with discount
            z_target = self.reward_scale * rewards + (1. - terminals) * ( self.discount ** self.n_step) * target_z_values_at_nstep

        #back propagation using S_{t} values in critic networks
        #need next presume tau based on s_t '

        tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=self.fp)
        new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
            next_obs,
            reparameterize=True,
            return_log_prob=True,
        )
        next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, new_next_actions, fp=self.target_fp)

        for grad_step in range(200):
            z1_pred= self.zf1(obs, actions, tau_hat)
            z2_pred = self.zf2(obs, actions, tau_hat)
            zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
            zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)

            gt.stamp('preback_zf', unique=False)
            self.zf1_optimizer.zero_grad()
            zf1_loss.backward()
            self.zf1_optimizer.step()
            gt.stamp('backward_zf1', unique=False)
            self.zf2_optimizer.zero_grad()
            zf2_loss.backward()
            self.zf2_optimizer.step()
            gt.stamp('backward_zf2', unique=False)

            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(self.zf1, self.target_zf1, self.soft_target_tau)
                ptu.soft_update_from_to(self.zf2, self.target_zf2, self.soft_target_tau)
        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(obs, actions, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(obs, actions, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)
            gt.stamp('preback_fp', unique=False)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()
            gt.stamp('backward_fp', unique=False)
        """
        Update Policy
        """

        for grad_step_policy in range(200):
            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs,
                reparameterize=True,
                return_log_prob=True,
            )
            risk_param = self.risk_schedule(self._n_train_steps_total)
            if self.risk_type == 'VaR':
                tau_ = ptu.ones_like(rewards) * risk_param
                q1_new_actions = self.zf1(obs, new_actions, tau_)
                q2_new_actions = self.zf2(obs, new_actions, tau_)
            else:
                with torch.no_grad():
                    new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs, new_actions, fp=self.fp)
                z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
                z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)
                if self.risk_type in ['neutral', 'std']:
                    q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                    q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdims=True)
                    if self.risk_type == 'std':
                        q1_std = new_presum_tau * (z1_new_actions - q1_new_actions).pow(2)
                        q2_std = new_presum_tau * (z2_new_actions - q2_new_actions).pow(2)
                        q1_new_actions -= risk_param * q1_std.sum(dim=1, keepdims=True).sqrt()
                        q2_new_actions -= risk_param * q2_std.sum(dim=1, keepdims=True).sqrt()
                else:
                    with torch.no_grad():
                        risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                    q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                    q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)
            policy_loss = (alpha * log_pi - q_new_actions).mean()

            gt.stamp('preback_policy', unique=False)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
            self.policy_optimizer.step()
            gt.stamp('backward_policy', unique=False)
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_tau)

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:

            if self.tau_type == 'fqf':
                ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['ZF1 Loss'] = zf1_loss.item()
            self.eval_statistics['ZF2 Loss'] = zf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z1 Predictions',
                ptu.get_numpy(z1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z2 Predictions',
                ptu.get_numpy(z2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Targets',
                ptu.get_numpy(z_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1


    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        networks = [
            self.policy,
            self.target_policy,
            self.zf1,
            self.zf2,
            self.target_zf1,
            self.target_zf2,
        ]
        if self.tau_type == 'fqf':
            networks += [
                self.fp,
                self.target_fp,
            ]
        return networks

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy.state_dict(),
            target_policy=self.target_policy.state_dict(),
            zf1=self.zf1.state_dict(),
            zf2=self.zf2.state_dict(),
            target_zf1=self.target_zf1.state_dict(),
            target_zf2=self.target_zf2.state_dict(),
        )
        if self.tau_type == 'fqf':
            snapshot['fp'] = self.fp.state_dict()
            snapshot['target_fp'] = self.target_fp.state_dict()
        return snapshot
