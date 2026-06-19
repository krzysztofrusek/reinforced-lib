from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass, Array, PRNGKey, Scalar

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class DiscreteThompsonSamplingState(AgentState):
    """
    Container for the state of the discrete Thompson sampling agent [12]_.

    Attributes
    ----------
    alpha : Array
        The concentration parameter of the Dirichlet distribution.

    References
    ----------
    .. [12] Stephen Tu.  The Dirichlet-Multinomial and Dirichlet-Categorical models for Bayesian inference.
       https://stephentu.github.io/writeups/dirichlet-conjugate-prior.pdf
    """

    alpha: Array


class DiscreteThompsonSampling(BaseAgent):
    r"""
    Discrete Thompson sampling agent .  action with
    the highest expected value is selected.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}` .
    alpha : float
        See also ``DiscreteThompsonSamplingState`` for interpretation. :math:`\alpha > 0`.
    outcomes : Array
        An array of shape (n_outcomes) containing the outcomes.
    """

    def __init__(
            self,
            n_arms: int,
            alpha: Array,
            outcomes: Array

    ) -> None:
        assert np.all(alpha > 0)

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=self.n_arms, alpha=alpha, outcomes=outcomes))
        self.update = jax.jit(partial(self.update, outcomes=outcomes))
        self.sample = jax.jit(partial(self.sample, outcomes=outcomes))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), int),
        })
    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(
            key: PRNGKey,
            n_arms: int,
            alpha: Array,
            outcomes: Array
    ) -> DiscreteThompsonSamplingState:
        return DiscreteThompsonSamplingState(
            alpha=jnp.tile(alpha, (n_arms, 1))
        )

    @staticmethod
    def update(
            state: DiscreteThompsonSamplingState,
            key: PRNGKey,
            action: int,
            reward: Scalar,
            outcomes: Array
    ) -> DiscreteThompsonSamplingState:
        update = (reward==outcomes).astype(state.alpha.dtype)
        alpha = state.alpha.at[action].add(update)
        return DiscreteThompsonSamplingState(
            alpha=alpha
        )

    @staticmethod
    def sample(state: DiscreteThompsonSamplingState, key: PRNGKey, outcomes: Array) -> int:
        p = jax.random.dirichlet(key, state.alpha)
        expectation = p.dot(outcomes)

        return jnp.argmax(expectation)