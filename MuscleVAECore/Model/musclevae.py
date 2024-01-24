import random
from typing import List,Dict
from numpy import dtype
import torch
from torch import nn
import torch.distributions as D
from MuscleVAECore.Model.trajectory_collection import TrajectorCollector
from MuscleVAECore.Model.world_model import SimpleWorldModel
from MuscleVAECore.Utils.mpi_utils import gather_dict_ndarray
from MuscleVAECore.Utils.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from modules import *
from ..Utils.motion_utils import *
from ..Utils import pytorch_utils as ptu
import time
import sys
from MuscleVAECore.Utils.radam import RAdam
from mpi4py import MPI

from ..Env.muscle_env import MuscleEnv

import VclSimuBackend
try:
    from VclSimuBackend.pymotionlib.PyTorchMotionData import PyTorchMotionData
except ImportError:
    PyTorchMotionData = VclSimuBackend.pymotionlib.PyTorchMotionData


mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# whether this process should do tasks such as trajectory collection....
# it's true when it's not root process or there is only root process (no subprocess)
should_do_subprocess_task = mpi_rank > 0 or mpi_world_size == 1

class MuscleVAE(nn.Module):
    """
    A ContorlVAE agent which includes encoder, decoder and world model
    """
    def __init__(self, observation_size, rigid_observation_sz, action_size, delta_size, env, **kargs):
        super().__init__()
        self.env_step_mode = kargs['env_step_mode']
        self.encoder = SimpleLearnablePriorEncoder(
            input_size= observation_size,
            condition_size= rigid_observation_sz,
            output_size= kargs['latent_size'],
            fix_var = kargs['encoder_fix_var'],
            **kargs).to(ptu.device)
        self.agent = GatingMixedDecoder(
            condition_size= observation_size,
            output_size=action_size,
            **kargs
        ).to(ptu.device)
        
        # statistics, will be used to normalize each dimention of observation
        statistics = env.stastics
        self.obs_mean = nn.Parameter(ptu.from_numpy(statistics['obs_mean']), requires_grad = False).to(ptu.device)
        self.obs_std = nn.Parameter(ptu.from_numpy(statistics['obs_std']), requires_grad= False).to(ptu.device)
        
        # world model
        muscle_state_size = observation_size-rigid_observation_sz
        self.world_model = SimpleWorldModel(rigid_observation_sz, action_size, delta_size, env, statistics, muscle_state_size,**kargs).to(ptu.device)
        
        # optimizer
        self.wm_optimizer = RAdam(self.world_model.parameters(), kargs['world_model_lr'], weight_decay=1e-3)
        ## add weight decay to prevent overfitting
        self.vae_optimizer = RAdam( list(self.encoder.parameters()) + list(self.agent.parameters()), kargs['musclevae_lr'], weight_decay=1e-3)        
        self.beta_scheduler = ptu.scheduler(0,8,0.009,0.09,500*8)
        
        # hyperparameters....
        self.action_sigma = 0.05
        self.max_iteration = kargs['max_iteration']
        self.collect_size = kargs['collect_size']
        self.sub_iter = kargs['sub_iter']
        self.save_period = kargs['save_period']
        self.evaluate_period = kargs['evaluate_period']
        self.world_model_rollout_length = kargs['world_model_rollout_length']
        self.musclevae_rollout_length = kargs['musclevae_rollout_length']
        self.world_model_batch_size = kargs['world_model_batch_size']
        self.musclevae_batch_size = kargs['musclevae_batch_size']
        
        # policy training weights                                    
        self.weight = {}
        for key,value in kargs.items():
            if 'musclevae_weight' in key:
                self.weight[key.replace('musclevae_weight_','')] = value
        
        # for real trajectory collection
        self.runner = TrajectorCollector(venv = env, actor = self, runner_with_noise = True)
        self.env : MuscleEnv  = env    
        self.use_muscle_state = self.env.use_muscle_state
        self.replay_buffer = ReplayBuffer(self.replay_buffer_keys(), kargs['replay_buffer_size']) if mpi_rank ==0 else None
        self.kargs = kargs

        ## fatigue muscle state
        self.use_coact = self.env.use_coact
        self.use_muscle = self.env.use_muscle
        self.use_scaled_muscle_len_rwd = self.env.use_scaled_muscle_len_rwd
        self.use_muscle_damping = self.env.use_muscle_damping
        
        self.simu_fps = self.env.simu_fps
        self.ctrl_fps = self.env.fps

        self.train_begin_iter = kargs['train_begin_iter']
        self.train_iter = self.train_begin_iter


    #--------------------------------for MPI sync------------------------------------#
    def parameters_for_sample(self):
        '''
        this part will be synced using mpi for sampling, world model is not necessary
        '''
        return {
            'encoder': self.encoder.state_dict(),
            'agent': self.agent.state_dict()
        }
    def load_parameters_for_sample(self, dict):
        self.encoder.load_state_dict(dict['encoder'])
        self.agent.load_state_dict(dict['agent'])
    
    #-----------------------------for replay buffer-----------------------------------#
    def world_model_data_name(self):
        return ['state', 'action', 'muscle_state']
    
    def policy_data_name(self):
        return ['state', 'target', 'target_scaled_muscle_len', 'muscle_state']
    
    def replay_buffer_keys(self):
        return ['state', 'action', 'target', 'target_scaled_muscle_len', 'muscle_state']

    #----------------------------for training-----------------------------------------#
    def train_one_step(self):
        
        time1 = time.perf_counter()
        
        # data used for training world model
        name_list = self.world_model_data_name()
        rollout_length = self.world_model_rollout_length
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length+1, # needs additional one state...
                            self.world_model_batch_size, 
                            self.sub_iter)
        for batch in  data_loader:
            world_model_log = self.train_world_model(*batch)
        
        time2 = time.perf_counter()
        
        # data used for training policy
        name_list = self.policy_data_name()
        rollout_length = self.musclevae_rollout_length
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length, 
                            self.musclevae_batch_size, 
                            self.sub_iter)
        for batch in data_loader:
            policy_log = self.train_policy(*batch)
        
        # log training time...
        time3 = time.perf_counter()      
        world_model_log['training_time'] = (time2-time1)
        policy_log['training_time'] = (time3-time2)
        
        # merge the training log...
        return self.merge_dict([world_model_log, policy_log], ['WM','Policy'])
    
    def mpi_sync(self):
        # sample trajectories
        if should_do_subprocess_task:
            with torch.no_grad():
                path : dict = self.runner.trajectory_sampling( math.floor(self.collect_size/max(1, mpi_world_size -1)), self )
                if self.env.balance:
                    self.env.update_val(path['done'], path['rwd'], path['frame_num'])
        else:
            path = {}

        tmp = np.zeros_like(self.env.val)
        mpi_comm.Allreduce(self.env.val, tmp)        
        self.env.val = tmp / mpi_world_size
        self.env.update_p()
        res = gather_dict_ndarray(path)
        if mpi_rank == 0:
            paramter = self.parameters_for_sample()
            mpi_comm.bcast(paramter, root = 0)
            self.replay_buffer.add_trajectory(res)
            info = {
                'rwd_mean': np.mean(res['rwd']),
                'rwd_std': np.std(res['rwd']),
                'episode_length': len(res['rwd'])/(res['done']!=0).sum()
            }
        else:
            paramter = mpi_comm.bcast(None, root = 0)
            self.load_parameters_for_sample(paramter)    
            info = None
        return info
    
    
    def train_loop(self):
        """training loop, MPI included
        """
        for i in range(self.train_begin_iter, self.max_iteration):            
            info = self.mpi_sync() 
            
            if mpi_rank == 0:
                print(f"----------training {self.train_iter} step--------")
                sys.stdout.flush()
                log = self.train_one_step()   
                log.update(info)       
                self.try_save(i)
                self.try_log(log, i)

            if should_do_subprocess_task:
                self.try_evaluate(i)

            self.train_iter += 1
                
    # -----------------------------------for logging----------------------------------#
    @property
    def dir_prefix(self):
        return 'Experiment'
    
    @property
    def bvh_prefix(self):
        return 'BVHFile'
    
    def save_before_train(self, args):
        """build directories for log and save
        """
        import os, time, yaml
        time_now = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
        dir_name = args['experiment_name']+'_'+time_now
        dir_name = mpi_comm.bcast(dir_name, root = 0)
        
        self.log_dir_name = os.path.join(self.dir_prefix,'log',dir_name)
        self.data_dir_name = os.path.join(self.dir_prefix,'checkpoint',dir_name)
        if mpi_rank == 0:
            os.makedirs(self.log_dir_name)
            os.makedirs(self.data_dir_name)

        mpi_comm.barrier()
        if mpi_rank > 0:
            f = open(os.path.join(self.log_dir_name,f'mpi_log_{mpi_rank}.txt'),'w')
            sys.stdout = f
            return
        else:
            yaml.safe_dump(args, open(os.path.join(self.data_dir_name,'config.yml'),'w'))
            self.logger = SummaryWriter(self.log_dir_name)
            
    def try_evaluate(self, iteration):
        if iteration % self.evaluate_period == 0:
            bvh_saver = self.runner.eval_one_trajectory(self)
            bvh_saver.to_file(os.path.join(self.data_dir_name,f'{iteration}_{mpi_rank}.bvh'))
        pass    

    def try_save_bvh(self, dir_name):
        import os
        bvh_saver = self.runner.eval_one_trajectory(self)
        data_dir_name = os.path.join(self.bvh_prefix,dir_name)
        bvh_saver.to_file(data_dir_name)
        
    
    def try_save(self, iteration):
        check_point = {
            'self': self.state_dict(),
            'wm_optim': self.wm_optimizer.state_dict(),
            'vae_optim': self.vae_optimizer.state_dict(),
            'balance': self.env.val,
            'train_iter':self.train_iter
        }
        torch.save(check_point, os.path.join(self.data_dir_name,f'latest.data'))
        if iteration % self.save_period ==0:
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))
    
    def try_load(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        self.load_state_dict(data['self'], strict = False)
        if isinstance(data.get("train_iter"), dict):
            self.train_begin_iter = data['train_iter']
        return data
        
    def try_log(self, log, iteration):
        for key, value in log.items():
            self.logger.add_scalar(key, value, iteration)
        self.logger.flush()
    
    def cal_rwd(self, **obs_info):
        observation = obs_info['observation_rigid']
        
        target = obs_info['target']
        error = pose_err(torch.from_numpy(observation), torch.from_numpy(target), self.weight, dt = self.env.dt)
        error = sum(error).item()
        return np.exp(-error/20)
    
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--latent_size", type = int, default = 64, help = "dim of latent space")
        arg_parser.add_argument("--max_iteration", type = int, default = 20001, help = "iteration for musclevae training")
        arg_parser.add_argument("--collect_size", type = int, default = 2048, help = "number of transition collect for each iteration")
        arg_parser.add_argument("--sub_iter", type = int, default = 8, help = "num of batch in each iteration")
        arg_parser.add_argument("--save_period", type = int, default = 2000, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--evaluate_period", type = int, default = 100, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--replay_buffer_size", type = int, default = 50000, help = "buffer size of replay buffer")

        return arg_parser
    
    #--------------------------API for encode and decode------------------------------#
    
    def encode(self, normalized_obs, normalized_target, **kargs):
        """encode observation and target into posterior distribution

        Args:
            normalized_obs (Optional[Tensor,np.ndarray]): normalized current observation
            normalized_target (Optional[Tensor, np.ndarray]): normalized current target 

        Returns:
            Tuple(tensor, tensor, tensor): 
                latent coder, mean of prior distribution, mean of posterior distribution 
        """
        return self.encoder(normalized_obs, normalized_target)
    
    def decode(self, normalized_obs, latent, **kargs):
        """decode latent code into action space

        Args:
            normalized_obs (tensor): normalized current observation
            latent (tensor): latent code

        Returns:
            tensor: action
        """
        action = self.agent(latent, normalized_obs)        
        return action
    
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        if self.use_muscle_state:
            observation = observation[:, :-15]
        return ptu.normalize(observation, self.obs_mean, self.obs_std)

    def normalize_target(self, target):
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if len(target.shape) == 1:
            target = target[None,...]
        
        return ptu.normalize(target, self.obs_mean, self.obs_std)
    
    def obsinfo2n_obs(self, obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation_rigid']
            else:
                observation = state2ob(obs_info['state'])
            n_observation = self.normalize_target(observation)
            if self.use_muscle_state:
                muscle_state = ptu.from_numpy(obs_info['muscle_state']).reshape(1, -1)
                n_observation = torch.cat([n_observation, muscle_state], dim=-1)
        return n_observation
    
    
    def act_tracking(self, **obs_info):
        """
        try to track reference motion
        used in trajectory collection
        """
        target = obs_info['target']
        n_target = self.normalize_target(target)
        n_observation = self.obsinfo2n_obs(obs_info)
        
        latent_code, mu_post, mu_prior = self.encode(n_observation, n_target)
        action = self.decode(n_observation, latent_code)

        info = {
            "latent":latent_code,
            "mu_prior": mu_prior,
            "mu_post": mu_post
        }
        return action, info
    
    
    def act_prior(self, obs_info):
        """
        try to track reference motion
        """
        n_observation = self.obsinfo2n_obs(obs_info)
        latent_code, mu_prior, logvar = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent_code)
        
        return action
    
    def act_determinastic(self, obs_info):
        action, _ = self.act_tracking(**obs_info)
        return action
                
    def act_distribution(self, obs_info):
        """
        Add noise to the output action
        """
        action = self.act_determinastic(obs_info)
        action_distribution = D.Independent(D.Normal(action, self.action_sigma), -1)
        return action_distribution
    
    #--------------------------------------Utils--------------------------------------#
    @staticmethod
    def merge_dict(dict_list: List[dict], prefix: List[str]):
        """Merge dict with prefix, used in merge logs from different model

        Args:
            dict_list (List[dict]): different logs
            prefix (List[str]): prefix you hope to add before keys
        """
        res = {}
        for dic, prefix in zip(dict_list, prefix):
            for key, value in dic.items():
                res[prefix+'_'+key] = value
        return res
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def try_load_world_model(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        wm_dict = data['self']
        wm_dict = {k.replace('world_model.',''):v for k,v in wm_dict.items() if 'world_model' in k}
        self.world_model.load_state_dict(wm_dict)
        return data

    
    