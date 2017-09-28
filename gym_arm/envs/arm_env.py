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
    init_step = 0.5
    self.state = self.np_random.uniform(np.pi-init_step, np.pi, 1) * [-1, 1][np.random.randint(1)]
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
    state_radius = 150
    r = 10

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)

      # state line
      self.state_line = rendering.make_circle(state_radius, filled=False)
      self.state_line.add_attr(rendering.Transform(translation=world_center))
      self.viewer.add_geom(self.state_line)

      # center circle
      self.center_line = rendering.make_circle(r)
      self.center_line.add_attr(rendering.Transform(translation=world_center))
      self.viewer.add_geom(self.center_line)

      # goal circle
      self.goal_circle = rendering.make_circle(r)
      self.goal_circle.set_color(255, 0, 0)
      self.goal_circle.add_attr(rendering.Transform(translation=(
        state_radius * np.cos(self.goal_position) + world_center[0],
        state_radius * np.sin(self.goal_position) + world_center[1]
      )))
      self.viewer.add_geom(self.goal_circle)
      
      # agent circle
      self.agent = rendering.Transform()
      agent_arm = rendering.make_polyline([(0, 0), (state_radius, 0)])
      agent_arm.add_attr(self.agent)
      agent_arm.set_linewidth(4)
      self.viewer.add_geom(agent_arm)
      agent_circle = rendering.make_circle(r)
      agent_circle.add_attr(rendering.Transform(translation=(state_radius, 0)))
      agent_circle.set_color(0, 255, 0)
      agent_circle.add_attr(self.agent)
      self.viewer.add_geom(agent_circle)

    pos = self.state[0]
    self.agent.set_translation(world_center[0], world_center[1])
    self.agent.set_rotation(pos)


    return self.viewer.render(return_rgb_array = mode=='rgb_array')


