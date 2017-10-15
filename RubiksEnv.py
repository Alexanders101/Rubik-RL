import numpy as np
from gym.core import Env
from gym import spaces

import Rotations

class RubiksEnv(Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, reward_func, scrambles=2, diff_reward=False, max_step=60):
        self._scrambles = scrambles
        self._reward_func = reward_func
        self._diff_reward = diff_reward
        self._max_step = max_step
        self._num_steps = 0

        self._cube = self._make_solved_cube()
        self._base_cube = self._make_solved_cube()


        # self._rotations = [Rotations.R, Rotations.R2, Rotations.R_,
        #                    Rotations.L, Rotations.L2, Rotations.L_,
        #                    Rotations.F, Rotations.F2, Rotations.F_,
        #                    Rotations.B, Rotations.B2, Rotations.B_,
        #                    Rotations.T, Rotations.T2, Rotations.T_,
        #                    Rotations.D, Rotations.D2, Rotations.D_,]

        self._rotations = [Rotations.R, Rotations.R_,
                           Rotations.L, Rotations.L_,
                           Rotations.F, Rotations.F_,
                           Rotations.B, Rotations.B_,
                           Rotations.T, Rotations.T_,
                           Rotations.D, Rotations.D_, ]

        self.action_space = spaces.Discrete(len(self._rotations))
        self.observation_space = spaces.Box(low=0, high=6, shape=(3, 3, 3, 3))


    def _step(self, action):
        reward = 0.0
        action = self._rotations[action]
        if (self._diff_reward):
            reward -= self._reward_func(self._cube)

        action(self._cube)

        reward += self._reward_func(self._cube)

        done = np.all(self._cube == self._base_cube)

        if (self._num_steps > self._max_step):
            done = True

        self._num_steps += 1

        return self._cube.copy(), reward, done, {}


    def _reset(self):
        self._cube = self._make_solved_cube()
        self._num_steps = 0
        for rot in np.random.randint(0, len(self._rotations), size=self._scrambles):
            self._rotations[rot](self._cube)

        return self._cube.copy()

    def _render(self, mode='human', close=False):
        return

    def _seed(self, seed=None):
        np.random.seed(seed)

    def _make_solved_cube(self):
        cube = np.zeros(shape=(3, 3, 3, 3), dtype=np.uint8)
        cube[:, :, 0, 2] = 1  # Front Face
        cube[:, :, 2, 2] = 6  # Back Face
        cube[:, 2, :, 1] = 2  # Top Face
        cube[:, 0, :, 1] = 4  # Bottom Face
        cube[2, :, :, 0] = 3  # Right Face
        cube[0, :, :, 0] = 5  # Left Face

        return cube

