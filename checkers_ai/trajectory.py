from checkers_ai.config import *


class Trajectory:

    def __init__(self):
        self.states = []
        self.softmax = []
        self.rewards = []

    def _feat2rwd(self, state):
        """
        :param state: board state
        :return: scalar reward from state features
        """
        chkr_diff = np.sum(np.argwhere(state == P1_CHKR), keepdims=False).astype(np.float32) - \
                    np.sum(np.argwhere(state == P2_CHKR), keepdims=False).astype(np.float32)
        king_diff = np.sum(np.argwhere(state == P1_KING), keepdims=False).astype(np.float32) - \
                    np.sum(np.argwhere(state == P2_KING), keepdims=False).astype(np.float32)
        rwd = 2 * (0.4 * chkr_diff + 0.6 * king_diff) / MAX_MOVES
        return rwd or 2.0 / MAX_MOVES

    def _bellman(self, step):
        if step < len(self.states) - 1:
            return self.rewards[step] + GAMMA * self._bellman(step + 1)
        else:
            return self.rewards[step]

    def update(self, state=None, softmax=None, terminal_rwd=None):
        """
        :param state: board state
        :param softmax: softmax over actions given board state
        :param terminal_rwd: terminal reward (None for non-terminal steps)
        :return: None
        """
        if state is not None:
            self.states.append(state)
        if softmax is not None:
            self.softmax.append(softmax)
        self.rewards.append(terminal_rwd or self._feat2rwd(state))

    def discounted_rewards(self, step:int):
        """
        :param step (step number)
        Compute reward function from step=step according to vanilla Bellman Eq:
            R_t = sum(k=step:step_terminal) GAMMA * R_step+1
            where R_t_terminal \in (REWARD_WIN, REWARD_DRAW, REWARD_LOSS)
        """
        return self._bellman(step=step)
