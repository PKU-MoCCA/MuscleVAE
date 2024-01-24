from MuscleVAECore.Model.musclevae import MuscleVAE
from MuscleVAECore.Utils.misc import load_data, load_yaml
import MuscleVAECore.Utils.pytorch_utils as ptu
from MuscleVAECore.Utils.motion_utils import resample_state
from MuscleVAECore.Env.muscle_env import MuscleEnv
import argparse
from direct.task import Task
import torch
import time
import numpy as np
class RandomPlayground(MuscleVAE):
    def __init__(self, observation_size, observation_rigid_size, action_size, delta_size, env, **kargs):
        super(RandomPlayground, self).__init__(observation_size, observation_rigid_size, action_size, delta_size, env, **kargs)
        self.mode = kargs['mode']        
        self.observation = self.env.reset()[0]
        
        self.step_generator = None
        self.other_objects = {}
        
        self.cnt = 0
        self.env.reset(0)
        
    
    def get_action(self, **obs_info):
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, logvar  = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent)
    
        info = {'mu': mu, 'logvar': logvar}
        return action, info
    
    def get_generator(self):
        action, info = self.get_action(**self.observation)
        action = ptu.to_numpy(action)
        return self.env.step_core(action, using_yield = True)
    
    
    def yield_step(self):
        if self.step_generator is None:
            self.step_generator = self.get_generator()  
        try:
            self.step_generator.__next__()
        except StopIteration as e:
            self.observation = e.value[0]
            self.step_generator = self.get_generator() 
            self.step_generator.__next__()
   
    
    def run(self, start_frame):
        self.env.reset(start_frame) 
        
        if self.mode == 'drawstuff':
            cnt = 0
            while True:
               
                self.yield_step()
                time.sleep(1/(120)) 
                cnt +=1
        else:
            raise NotImplementedError
        
    
    def after_step(self, server_scene):
        character = server_scene.characters[1]   
        pass
    
      
    @staticmethod
    def build_arg(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument('--mode', default='drawstuff', type=str)
        parser.add_argument('--show', default=True, action='store_true')
        parser.add_argument('--drawstuff_render_mode', default=1, type=int, help='render mode, 0 means drawing muscles and capsules, 1 means drawing muscles only')
        parser.add_argument("--env_fps", type=int, default=20, help="fps of control policy")
        parser.add_argument("--play_random_mode", type=int, default=1, help="if true, transition to next random")
        parser.add_argument('--start_frame', type=int, default=0)
        
        parser = MuscleEnv.add_specific_args(parser)
        parser = MuscleVAE.add_specific_args(parser)

        args = vars(parser.parse_args())
        config =load_yaml(initialdir ='Data/NNModel')
        
        for key in config:
            if key not in args or args[key] is None:
                args[key] = config[key]

        ptu.init_gpu(True)
        return args


if __name__ == '__main__':
    args = RandomPlayground.build_arg()
    env = MuscleEnv(**args)
    rigid_observation_sz = 371
    muscle_observation_sz =  5 * 3
    time_observation_sz = args['time_observation_sz']
    observation_sz = rigid_observation_sz + muscle_observation_sz + time_observation_sz
    muscle_action = 284
    action_sz = muscle_action 
    playground = RandomPlayground(observation_sz, rigid_observation_sz, action_sz, 138, env, **args)
    
    print("in random ground 'experiment_name'",args['experiment_name'])
    print("in random ground 'env_step_mode'",args['env_step_mode'])


    import tkinter.filedialog as fd
    data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    playground.try_load(data_file)
    playground.run(args['start_frame'])