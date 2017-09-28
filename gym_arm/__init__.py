from gym.envs.registration import register

register(
  id='arm-v0',
  entry_point='gym_arm.envs:ArmEnv',
)

register(
  id='arm_noise-v0',
  entry_point='gym_arm.envs:ArmEnvNoise',
)
