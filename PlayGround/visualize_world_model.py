import numpy as np
import torch

from MuscleVAECore.Env.muscle_env import MuscleEnv
from MuscleVAECore.Env.muscle_env import character_state, character_muscle_state, state2ob
from MuscleVAECore.Utils import pytorch_utils as ptu
from PlayGround.track_something import TrackPlayground, RandomPlayground
import matplotlib.pyplot as plt
import argparse
import time

class WorldModelVisuEnv(MuscleEnv):
    def normalize_target(self, target, vae):
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if len(target.shape) == 1:
            target = target[None,...]
        
        return ptu.normalize(target, vae.obs_mean, vae.obs_std)
    

    def step_core(self, action, vae, using_yield=False, **kargs):
        ref_muscle_len  = self.get_ref_muscle_length(action).reshape(-1)
        # world model visualiztion (using ref-character)
        last_state = character_state(self.sim_character, self.state if self.recompute_velocity else None, self.dt)
        last_state_new = torch.from_numpy(last_state[np.newaxis, :, :]).cuda()
        last_muscle_state = character_muscle_state(self.sim_character)
        last_muscle_state_new = torch.from_numpy(last_muscle_state[np.newaxis,  :]).cuda()
        last_obs = state2ob(last_state_new)
        n_obs = self.normalize_target(last_obs, vae)
        n_obs = torch.cat([n_obs, last_muscle_state_new], dim=-1)
        torch_action = torch.from_numpy(action).cuda()
        world_model_state, _ = vae.world_model(last_state_new, torch_action, last_muscle_state_new, n_observation=n_obs)
        
        modified_ref_state = world_model_state.detach().cpu().numpy()[0, :, :]
        modified_body_pos = self.sim_char_root_body_com[ [0,2]] - modified_ref_state[0, [0,2]] +[-1.5,  0.0]
        modified_ref_state[:, [0,2]] = modified_ref_state[:, [0,2]] + modified_body_pos
        self.load_character_state(self.ref_character, modified_ref_state)
        
        for i in range(self.substep):
            sim_muscle_len = self.get_sim_muscle_length()
            anchor_full, muscle_amp , passive_amp, unclip_muscle_force= self.apply_pd_muscle_force(sim_muscle_len, ref_muscle_len)
            activation, unclip_activation = self.calc_hill_type_activation( muscle_amp, sim_muscle_len)
            self.sim_character.activation_levels = activation 
            self.apply_fatigue()
            self.sim_character.pd_muscle_amp = muscle_amp
            self.post_step_draw(muscle_amp, sim_muscle_len)
            self.rigid_body_simulate()
            self.sim_char_root_body_com = self.sim_character.bodies[0].PositionNumpy

            if using_yield:
                yield self.sim_character.save()
        
        self.state = character_state(self.sim_character, self.state if self.recompute_velocity else None, self.dt)
        self.observation = state2ob(torch.from_numpy(self.state)).numpy()
        self.observation_rigid = self.observation
        self.muscle_state = np.zeros(15, dtype=np.float64)
        if self.use_muscle_state:
            self.muscle_state = character_muscle_state(self.sim_character)
            self.observation = np.concatenate((self.observation.flatten(), self.muscle_state.flatten()))
            
        reward = 0 
        
        done = self.cal_done(self.state, self.observation_rigid)
        info = self.get_info()
        self.after_step()
        
        observation = {
            'state': self.state,
            'muscle_state': self.muscle_state ,
            'observation_rigid': self.observation_rigid,
            'observation': self.observation,
            'target': self.target,
            'target_scaled_muscle_len': self.target_scaled_muscle_len,
        }

        if not using_yield:
            yield observation, reward, done, info
        else:
            return observation, reward, done, info  
        
    def after_step(self, **kargs):
        test_counter = self.counter+1
        if self.play_random_mode:
            if test_counter>= len(self.motion_data.done) or self.motion_data.done[test_counter] :
                self.step_counter(random = True)
            elif (self.step_cnt % self.random_count)==0 and self.step_cnt!=0:
                self.step_counter(random = True)
            else:
                self.step_counter(random = False)
        else:
            if test_counter>= len(self.motion_data.done) or self.motion_data.done[test_counter] :
                self.step_counter(random = True)
            else:
                self.step_counter(random = False)
        
        # do not modify ref-character

        self.target = self.get_target()
        self.target_scaled_muscle_len = self.get_target_scaled_muscle_len()
        self.step_cnt += 1

class TrackPlaygroundModified(TrackPlayground):
    def __init__(self, observation_size, observation_rigid_size, action_size, delta_size, env, **kargs):
        super().__init__(observation_size, observation_rigid_size, action_size, delta_size, env, **kargs)

    def get_generator(self):
        action, info = self.get_action(**self.observation)
        action = ptu.to_numpy(action)
        return self.env.step_core(action, self, using_yield = True)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = RandomPlayground.build_arg(parser)
    args['play_random_mode'] = 0
    env = WorldModelVisuEnv(**args)
    rigid_observation_sz = 371
    muscle_observation_sz =  5 * 3
    observation_sz = rigid_observation_sz + muscle_observation_sz 
    muscle_action = 284
    action_sz = muscle_action 
    playground = TrackPlaygroundModified(observation_sz, rigid_observation_sz, action_sz, 138, env, **args)
    
    print("in visu world model 'experiment_name'",args['experiment_name'])
    print("in visu world model 'env_step_mode'",args['env_step_mode'])

    import tkinter.filedialog as fd
    data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    playground.try_load(data_file)
    
    playground.run(args['start_frame'])