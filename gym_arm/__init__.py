from gym.envs.registration import register

register(
  id='arm-v0',
  entry_point='gym_arm.envs:ArmEnv',
)
