from copy import deepcopy
from typing import Any, Dict, Optional, Union, Callable

import numpy as np
import torch
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy
from torch.nn.utils.rnn import pad_sequence
import random

class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        subgoal_detector:torch.nn.Module,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        subgoal_input_length:int = 100,
        clip_subgoal:int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.subgoal_input_length = subgoal_input_length
        self.subgoal_detector = subgoal_detector
        self.subgoal_detector.cuda()
        self.clip_subgoal = clip_subgoal
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad
        self._compile()
    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray, subgoal:np.ndarray, rand_index:np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_subgoal_nstep_return(
            batch,
            buffer,
            indices,
            subgoal,
            rand_index,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm,
        )
        return batch

    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        if buffer is None:
            return {}
        batch, indices = buffer.sample(1)
        while True:
            if indices[-1] == buffer.next([indices[-1]])[0]:
                batch_new, indices_new = buffer.sample(1)
                indices = np.append(indices,indices_new)
            else:
                indices = np.append(indices,buffer.next([indices[-1]]))
            if len(indices) > self.subgoal_input_length-1:
                break
        batch_for_subgoal = buffer[indices]
        obs = batch_for_subgoal['obs_next']
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        input_for_subgoal = obs_next / 255.
        input_for_subgoal = torch.tensor([input_for_subgoal])
        off_sets = torch.tensor([self.subgoal_input_length])
        input_for_subgoal = pad_sequence(input_for_subgoal, batch_first=True, padding_value=0)
        input_for_subgoal = input_for_subgoal.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            mask = self.subgoal_detector(input_for_subgoal,off_sets)
        mask = mask.detach().cpu().numpy()[0]
        rand_index = random.sample(range(self.clip_subgoal,self.subgoal_input_length-self.clip_subgoal,1),sample_size)
        rand_index = np.array(rand_index)
        indices_sample =  indices[rand_index]
        batch = buffer[indices_sample]


        self.updating = True
        batch = self.process_fn(batch, buffer, indices_sample, mask, rand_index)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices_sample)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result

    def compute_subgoal_nstep_return(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices_l: np.ndarray,
        subgoal: np.ndarray,
        rand_index: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        target_q_list =[]
        target_q_torch_list = []
        n_step_global = n_step
        for ind_global, indice in enumerate(indices_l):
            indice = np.array([indice])
            bsz = len(indice)
            indices = [indice]
            subgoal_ = subgoal[rand_index[ind_global]:rand_index[ind_global]+n_step_global]
            n_step = np.argmax(subgoal_)+1
            for _ in range(n_step - 1):
                indices.append(buffer.next(indices[-1]))
            indices = np.stack(indices)
            # terminal indicates buffer indexes nstep after 'indice',
            # and are truncated at the end of each episode
            terminal = indices[-1]
            with torch.no_grad():
                target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
            target_q = to_numpy(target_q_torch.reshape(bsz, -1))
            target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
            end_flag = buffer.done.copy()
            end_flag[buffer.unfinished_index()] = True
            target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
            target_q_list.append(target_q)
            target_q_torch_list.append(target_q_torch)
        target_q_torch = torch.cat(target_q_torch_list,dim=0)
        target_q = np.concatenate(target_q_list,axis=0)
        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch


    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self._clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        _gae_return(f64, f64, f64, b, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = rew[now].reshape(bsz, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)
