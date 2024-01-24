import numpy as np
import torch
import operator
from ..Utils import pytorch_utils as ptu
from ..Env.muscle_env import MuscleEnv

class TrajectorCollector():
    def __init__(self, **kargs) -> None:
        self.reset(**kargs)
    
    def reset(self, venv, **kargs):
        self.env: MuscleEnv = venv
        self.with_noise = kargs['runner_with_noise']
        
        
    def trajectory_sampling(self, sample_size, actor):
        cnt = 0
        res = []
        while cnt < sample_size:
            trajectory =self.sample_one_trajectory(actor) 
            res.append(trajectory)
            cnt+= len(trajectory['done'])
        
        res_dict = {}
        for key in res[0].keys():
            res_dict[key] = np.concatenate( list(map(operator.itemgetter(key), res)) , axis = 0)
        return res_dict
    
    def eval_one_trajectory(self, actor):
        saver = self.env.get_bvh_saver()
        observation, info = self.env.reset()
        while True: 
            saver.append_no_root_to_buffer()

            # when eval, we do not hope noise...
            action = actor.act_determinastic(observation)
            action = ptu.to_numpy(action).flatten()
            new_observation, rwd, done, info = self.env.step(action)
            observation = new_observation
            if done:
                break
        return saver
    
    def sample_one_trajectory(self, actor):
        observation, info = self.env.reset()    
        states, muscle_states,  targets, target_scaled_muscle_lens, actions, rwds, dones, frame_nums = [[] for i in range(8)]
         
        while True: 
            if np.isnan(observation['observation']).any():
                states, muscle_states,  targets, target_scaled_muscle_lens, actions, rwds, dones, frame_nums = [[] for i in range(8)]
                observation, info = self.env.reset()  
            
            if self.with_noise:
                action_distribution = actor.act_distribution(observation)
                action = action_distribution.sample()
            else:
                action = actor.act_determinastic(observation)
            
            if np.random.choice([True, False], p = [0.4, 0.6]):
                action = actor.act_prior(observation)
                action = action + torch.randn_like(action) * 0.05
            action = ptu.to_numpy(action).flatten()
            
            states.append(observation['state'])
            muscle_states.append(observation['muscle_state'])
            actions.append(action)
            targets.append(observation['target'])
            target_scaled_muscle_lens.append(observation['target_scaled_muscle_len'])
            
            new_observation, rwd, done, info = self.env.step(action)
            
            rwd = actor.cal_rwd(observation_rigid = new_observation['observation_rigid'], target = new_observation['target'])
            rwds.append(rwd)
            dones.append(done)
            frame_nums.append(info['frame_num'])
            
            observation = new_observation
            if done:
                break
        output_dict = {
            'state': states,
            'action': actions,
            'muscle_state': muscle_states,
            'target': targets,
            'target_scaled_muscle_len':target_scaled_muscle_lens,
            'done': dones,
            'rwd': rwds,
            'frame_num': frame_nums
        }
    
        return output_dict
            
            
            
            