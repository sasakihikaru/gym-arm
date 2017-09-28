import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class ArmEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
  }

  def __init__(self):
    self.min_position = -np.pi
    self.max_position = np.pi

    self.low  = np.array([self.min_position])
    self.high = np.array([self.max_position])
    
    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(self.low, self.high)

    self.viewer = None

    self._seed()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), '%r (%s) invalid' % (action, type(action))

    action_step = np.pi / 12

    state = self.state[0]
    state += (action - 1) * action_step

    if state > np.pi:
      state -= 2*np.pi
    if state < -np.pi:
      state += 2*np.pi

    self.state = np.array([state])

    reward = (np.pi - abs(self.state[0] - self.goal_position)) / np.pi

    done = False

    return self.state, reward, done, {}

  def _reset(self):
    init_step = 0.1
    self.state = self.np_random.uniform(np.pi-init_step, np.pi+init_step, 1)
    self.goal_position = self.np_random.uniform(low=-1*init_step, high=init_step)
    return self.state

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400
    
    world_center = (screen_width/2, screen_height/2)

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)

      radius = 150
      ts = np.linspace(0, 2*np.pi, 100)
      xs = radius * np.cos(ts) + world_center[0]
      ys = radius * np.sin(ts) + world_center[1]
      xys = list(zip(xs, ys))

      self.track = rendering.make_polyline(xys)
      self.track.set_linewidth(2)
      self.viewer.add_geom(self.track)

    return self.viewer.render(return_rgb_array = mode=='rgb_array')


