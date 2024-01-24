
from MuscleVAECore.Env.muscle_env import MuscleEnv
from PlayGround.random_generation import RandomPlayground
import torch
import MuscleVAECore.Utils.pytorch_utils as ptu
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time


class TrackPlayground(RandomPlayground):

    def get_action(self, **obs_info):
        n_observation = self.obsinfo2n_obs(obs_info)
        target = obs_info['target']
        n_target = self.normalize_target(target)
        
        latent_code, mu_post, mu_prior = self.encode(n_observation, n_target)
        action = self.decode(n_observation, mu_post)

        info = {}
        return action, info
    
    def yield_step(self):
        super().yield_step()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = RandomPlayground.build_arg(parser)
    args['play_random_mode'] = 0
    env = MuscleEnv(**args)
    rigid_observation_sz = 371
    muscle_observation_sz =  5 * 3
    observation_sz = rigid_observation_sz + muscle_observation_sz 
    muscle_action = 284
    action_sz = muscle_action 
    playground = TrackPlayground(observation_sz, rigid_observation_sz, action_sz, 138, env, **args)
    
    print("in track ground 'experiment_name'",args['experiment_name'])
    print("in track ground 'env_step_mode'",args['env_step_mode'])

    import tkinter.filedialog as fd
    data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    playground.try_load(data_file)
    
    playground.run(args['start_frame'])
    