import argparse
from MuscleVAECore.Env.muscle_env import MuscleEnv
from MuscleVAECore.Model.musclevae import MuscleVAE
from MuscleVAECore.Utils.misc import load_data, load_yaml
from MuscleVAECore.Utils.motion_utils import state2ob
from MuscleVAECore.Utils.pytorch_utils import build_mlp
from MuscleVAECore.Utils.radam import RAdam
from PlayGround.playground_util import get_root_facing
from random_generation import RandomPlayground
import MuscleVAECore.Utils.pytorch_utils as ptu
import torch
import numpy as np
import types
from scipy.spatial.transform import Rotation
import psutil
import MuscleVAECore.Utils.pytorch_utils as ptu

from mpi4py import MPI
from collections import deque

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def random_target(env):
    speed = np.random.choice(env.speed_range)
    direction_angle = np.random.uniform(0, np.pi * 2)
    res = np.array([speed, direction_angle])
    return res

def speed_target(self):
    if not hasattr(self, 'target') or self.target is None:
        self.target = random_target(self)
    return self.target


def hand_control_1_minute(self):
    """
    In drawstuff mode, it is not easy to use the keyboard to control the character.
    """
    if self.interactor.time_step < 400:
        angle = 35.0*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 400 and  self.interactor.time_step < 800:
        angle =  60.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 800 and  self.interactor.time_step < 1200:
        angle =  75.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 1200 and  self.interactor.time_step < 1600:
        angle = 180*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 1600 and  self.interactor.time_step < 2000:
        angle = 180*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 2000 and  self.interactor.time_step < 2400:
        angle = 150*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 2400 and  self.interactor.time_step < 2800:
        angle = 0
        velo_norm = 3.0
    elif  self.interactor.time_step >= 2800 and  self.interactor.time_step < 3200:
        angle = 0
        velo_norm = 2.0
    elif  self.interactor.time_step >= 3200 and  self.interactor.time_step < 3600:
        angle = 60*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 3600 and  self.interactor.time_step < 4000:
        angle =  120.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 4000 and  self.interactor.time_step < 4400:
        angle =  75.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 4400 and  self.interactor.time_step < 4800:
        angle = 20*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 4800 and  self.interactor.time_step < 5200:
        angle = 40*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 5200 and  self.interactor.time_step < 5600:
        angle = 150*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 5600 and  self.interactor.time_step < 6000:
        angle = 0
        velo_norm = 3.0
    elif  self.interactor.time_step >= 6000 and  self.interactor.time_step < 6400:
        angle = 60*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 6400 and  self.interactor.time_step < 6800:
        angle = 60*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 6800 and  self.interactor.time_step < 7200:
        angle = 60*np.pi/180
        velo_norm = 2.0
    else:
        angle = -180.0*np.pi/180
        velo_norm = 3.0

    return angle, velo_norm


def after_step(self, **kargs):
    self.step_cnt += 1
    self.target_scaled_muscle_len = self.get_target_scaled_muscle_len()
    if not self.interaction:
        if self.step_cnt % self.random_count == 0:
            self.target = random_target(self)
    else:
        angle, velo_norm = hand_control_1_minute(self)
        res = np.array([velo_norm, angle])
        self.target = res



def after_substep(self):
    self.interactor.time_step += 1



class SpeedPlayground(RandomPlayground):
    def __init__(self,observation_size, observation_rigid_size, action_size, delta_size, env, **kargs):
        kargs['replay_buffer_size'] = 5000
        super().__init__(observation_size, observation_rigid_size, action_size, delta_size, env, **kargs)
        self.observation_size = observation_size
        self.latent_size = kargs['latent_size']
        self.batch_size = 512
        self.collect_size = 500
        self.env.max_length = 512
        self.runner.with_noise = kargs['train'] # use act_determinastic....
        self.time_step = 0
        
        if mpi_rank == 0:
            self.replay_buffer.reset_max_size(20000)
        
        self.show = kargs['show']
        self.env.speed_range = [0,0,1,2,3]
        self.env.get_target = types.MethodType(speed_target, self.env)
       
        self.env.interaction = kargs['interaction']
        self.env.after_step = types.MethodType(after_step, self.env)   
        self.env.after_substep = types.MethodType(after_substep, self.env)
        
        self.env.show = self.show
        self.build_high_level()
        
        self.mass = self.env.sim_character.body_info.mass_val / self.env.sim_character.body_info.sum_mass
        self.mass = ptu.from_numpy(self.mass).view(-1)

        self.dance = kargs['dance']
        self.show_arrow = True

        if self.show:
            if self.mode == 'drawstuff':
                try:
                    from VclSimuBackend.ODESim.Loader.MeshCharacterLoader import MeshCharacterLoader
                except ImportError:
                    import VclSimuBackend
                    MeshCharacterLoader = VclSimuBackend.ODESim.MeshCharacterLoader
                MeshLoader = MeshCharacterLoader(self.env.scene.world, self.env.scene.space)
                self.env.arrow = MeshLoader.load_from_obj('./Data/Misc/drawstuff/arrow.obj', 'arrow', volume_scale=1, density_scale=1)
                self.env.arrow.is_enable = False
                self.env.arrow = self.env.arrow.root_body
                self.env.base_rotation = Rotation.from_rotvec(np.array([-np.pi/2,0,0]))
                self.env.interactor = self
                



    #-----------------------------deal with parameters--------------------------------#
    def parameters_for_sample(self):
        res =  super().parameters_for_sample()
        res['high_level'] = self.high_level.state_dict()
        return res
    
    def load_parameters_for_sample(self, dict):
        super().load_parameters_for_sample(dict)
        self.high_level.load_state_dict(dict['high_level'])
    
    def try_evaluate(self, iteration):
        pass
    
    def try_save(self, iteration):
        if iteration % self.save_period == 0:
            check_point = {
                    'self': self.state_dict(),
                    'wm_optim': self.wm_optimizer.state_dict(),
                    'vae_optim': self.vae_optimizer.state_dict(),
                    'balance': self.env.val,
                    'high_level': self.high_level.state_dict(),
                    'high_level_optim': self.high_level_optim.state_dict(),
                }
            import os
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))

    def try_load(self, data_file):
        data = super().try_load(data_file)
        self.high_level.load_state_dict(data['high_level'])
        return data
    @property
    def dir_prefix(self):
        return 'Experiment/playground'
    
    def cal_rwd(self, **obs_info):
        return 0
    
    #------------------------------------------task-----------------------------------#    
    @property   
    def task_ob_size(self):
        return self.observation_size + 3
    
    def build_high_level(self):
        self.high_level = build_mlp(self.task_ob_size, self.latent_size, 3, 256, 'ELU').to(ptu.device)
        self.high_level_optim = RAdam(self.high_level.parameters(), lr=1e-3)
        lr = lambda epoach: max(0.99**(epoach), 1e-1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.high_level_optim, lr)
        
    @staticmethod
    def target2n_target(state, target):
        if len(state.shape) ==2:
            state = state[None,...]
        if len(target.shape) ==1:
            target = target[None,...]
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if isinstance(state, np.ndarray):
            state = ptu.from_numpy(state)
        facing_direction = get_root_facing(state)
        facing_angle = torch.arctan2(facing_direction[:,2], facing_direction[:,0])
        delta_angle = target[:,1] - facing_angle
        res = torch.cat([target[:,0, None], torch.cos(delta_angle[:,None]), torch.sin(delta_angle[:,None])], dim = -1)
        return res
        
    
    #------------------------------------------acting-------------------------------#
    def act_task(self, **obs_info):
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, _ = self.encoder.encode_prior(n_observation)    
        n_target = self.target2n_target(obs_info['state'], obs_info['target'])
        
        task = torch.cat([n_observation, n_target], dim=1)
        offset = self.high_level(task)
        if self.dance:
            if n_target[...,2].abs()<0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu+offset
        
        action = self.decode(n_observation, latent)
        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset
        }
    
    def act_determinastic(self, obs_info):
        return self.act_task(**obs_info)[0]
    
    
    
    #------------------------------------------playing-------------------------------#
    def get_action(self, **obs_info):
        return self.act_task(**obs_info)
    

    #-----------------------------------------configuring----------------------------#
    @staticmethod
    def build_arg(parser = None):
        import yaml
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default = 'drawstuff', type = str)
        parser.add_argument('--show', default = False, action='store_true')
        parser.add_argument('--drawstuff_render_mode', default = 1, type=int)
        parser = MuscleEnv.add_specific_args(parser)
        parser = MuscleVAE.add_specific_args(parser)
        args = vars(parser.parse_args())
        config = load_yaml(initialdir ='Data/NNModel/Pretrained')
        args.update(config)

        return args
       
        
if __name__ == "__main__":
    if mpi_rank ==0:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', default=0, type = int)
        parser.add_argument('--interaction', default=True,  action='store_true')
        parser.add_argument("--use_as_highlevel", type=int, default=1, help = "whether use highlevel train")
        parser.add_argument("--env_step_mode", type=str, default='fatigue_dynamics_constrain', help = "step mode in env")
        parser.add_argument('--gpu', type = int, default=0, help='gpu id')
        parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
        parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
        parser.add_argument('--experiment_name', type = str, default="speed_playground", help="")
        parser.add_argument('--muscle_capacity_ratio', type = float, default=1.0, help='how muscle muscle capacity of your model')
        parser.add_argument('--muscle_max_force_at', type = float, default=1.0, help='how muscle muscle force could be')
    
        args = SpeedPlayground.build_arg(parser)
        import tkinter.filedialog as fd
        data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    args = mpi_comm.bcast(None if mpi_rank!=0 else args, root=0)
    ptu.init_gpu(True, gpu_id=args['gpu'])
    if args['cpu_e'] !=-1:
        p = psutil.Process()
        cpu_lst = p.cpu_affinity()
        try:
            p.cpu_affinity(range(args['cpu_b'],args['cpu_e']))   
        except:
            pass 

    data_file = mpi_comm.bcast(None if mpi_rank!=0 else data_file, root=0)
    env = MuscleEnv(show_drawstuff=False,**args)
    
    rigid_observation_sz = 371
    muscle_observation_sz =  5 * 3
    observation_sz = rigid_observation_sz + muscle_observation_sz 
    muscle_action = 284
    joint_torque_action = 66
    action_sz = muscle_action 
    playground = SpeedPlayground(observation_sz, rigid_observation_sz, action_sz, 138, env, **args)
    
    
    print("in joystick ground 'env_step_mode'",args['env_step_mode'])
    print("in joystick ground 'experiment_name'",args['experiment_name'])
    playground.try_load(data_file)
    playground.run(0)