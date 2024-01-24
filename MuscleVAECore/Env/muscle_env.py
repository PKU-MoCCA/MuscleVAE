import pickle
import numpy as np
import torch
from ..Utils.motion_dataset import MotionDataSet
from scipy.spatial.transform import Rotation
try:
    from VclSimuBackend import SetInitSeed
except:
    from ModifyODE import SetInitSeed
from ..Utils.motion_utils import character_state, state2ob, state_to_BodyInfoState, character_muscle_state, state2bodypose_using_fk, state2bodypose, resample_state, resample_BodyInfoState , np_state2scaled_muscle_length
from ..Utils.index_counter import index_counter
from ..Utils import pytorch_utils as ptu
from ..Utils.diff_quat import *
from DrawStuffUtils.init_viewer import viewer_start
from Muscle.muscle_scene_loader import JsonSceneWithMuscleLoader

from collections import deque

import VclSimuBackend as ode
try:
    from VclSimuBackend.Common.MathHelper import MathHelper
    from VclSimuBackend.ODESim.Saver import CharacterToBVH
    from VclSimuBackend.ODESim.PDControler import DampedPDControler
    from VclSimuBackend.pymotionlib.PyTorchMotionData import PyTorchMotionData
except ImportError:
    MathHelper = VclSimuBackend.Common.MathHelper
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    DampedPDControler = VclSimuBackend.ODESim.PDController.DampedPDControler
    PyTorchMotionData = VclSimuBackend.pymotionlib.PyTorchMotionData



import datetime
current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y-%m-%d_%H_%M_%S")

class MuscleEnv():
    """A tracking environment, also performs as a environment base... because we need a step function.
    it contains:
        a ODEScene: physics environment which contains two characters, one for simulation another for kinematic
        a MotionDataset: contains target states and observations of reference motion
        a counter: represents the number of current target pose
        a step_cnt: represents the trajectory length, will be set to zero when env is reset
    """
    def __init__(self, **kargs) -> None:
        super(MuscleEnv, self).__init__()
        self.object_reset(**kargs)
        self.muscle_init(**kargs)

    def object_reset(self, **kargs):
        if 'seed' in kargs:
            self.seed(kargs['seed'])  
        self.mode = kargs['mode']
        self.show = kargs['show']
        scene_fname = kargs['env_scene_fname']
        SceneLoader = JsonSceneWithMuscleLoader()
        self.scene = SceneLoader.file_load(scene_fname)
        self.scene.characters[1].is_enable = False  
        self.scene.characters[1].anchor_positions = self.scene.characters[0].anchor_positions+[-150, 0.0, -150]
        self.scene.characters[1].set_render_color(np.array([174, 219, 206], dtype=np.float64)/256.0)
        self.scene.contact_type = kargs['env_contact_type']
        self.scene.self_collision = 0 
        self.fps = kargs['env_fps']
        self.substep = kargs['env_substep']
        self.simu_fps = self.fps * self.substep
        self.dt = 1/self.fps
        self.simu_dt = 1/self.simu_fps
        
        self.min_length = kargs['env_min_length']
        self.max_length = kargs['env_max_length']
        self.err_threshod = kargs['env_err_threshod']
        self.err_length = kargs['env_err_length']
        self.stable_pd = DampedPDControler(self.sim_character)
        name_list = self.sim_character.body_info.get_name_list()
        self.head_idx = name_list.index('Head')
        self.balance = not kargs['env_no_balance']
        self.use_com_height = kargs['env_use_com_height']
        self.recompute_velocity = kargs['env_recompute_velocity']
        self.random_count = kargs['env_random_count']
        self.step_mode = kargs['env_step_mode']
        self.motion_data : MotionDataSet = pickle.load(open(kargs['motion_dataset'],'rb'))
        self.frame_num = self.motion_data.state.shape[0]
        self.init_index = index_counter.calculate_feasible_index(self.motion_data.done,24+1) 
        self.begin_init_index = index_counter.calculate_beign_feasible_index(self.motion_data.done)
        if self.balance:
            self.val = np.zeros(self.frame_num)    
            self.update_p()
        
        return 
    
    def muscle_init(self, **kargs):
        self.use_muscle = 1 
        self.use_muscle_state = 1
        self.use_scaled_muscle_len_rwd = 1
        self.use_muscle_damping = 1 
        self.use_coact = 1 
        self.play_random_mode = kargs['play_random_mode']
        pose_aware_f0_max = './Data/Misc/MuscleVAE_force.txt'
        kp_coff_ratio = kargs['kp_coff_ratio']
        self.speed_range = [0,0,1,2,3]
        kp_coff = self.sim_character.muscle_properties[:, 0] * np.loadtxt(kp_coff_ratio, dtype=np.float64)
        self.numpy_kp_coff = kp_coff
        pose_aware_f0_max = np.loadtxt(pose_aware_f0_max, dtype=np.float64) * 1.5
        self.sim_character.set_simulate_fps(self.simu_fps)
        self.sim_character.adjust_muscle_kp_coff(kp_coff)
        self.sim_character.adjust_muscle_kd_coff(kp_coff)
        self.sim_character.adjust_pose_aware_f0_max(pose_aware_f0_max)
        self.sim_character.cython_build_coact_matrix_avg()        
        self.sim_char_root_body_com = self.sim_character.bodies[0].PositionNumpy
        self.fatigue_change_ratio = kargs['fatigue_adjust_ratio']
        self.sim_character.r_para = 2.0
        self.sim_character.F_para = 0.05 * self.fatigue_change_ratio
        self.sim_character.R_para = 0.01 * self.fatigue_change_ratio
        self.sim_character.LR_para = 50.0
        self.sim_character.LD_para = 50.0
        fatigue_part_calc_mode = 1
        self.sim_character.prepare_3cc_fatigue_model_with_input_mode(fatigue_part_calc_mode)
        self.env_fps = kargs['env_fps']
        self.env_substep = kargs['env_substep']
        self.no_fatigue_random_start = kargs['no_fatigue_random_start']
        self.pytorch_motion_data = PyTorchMotionData.PyTorchMotionData()
        self.pytorch_motion_data.build_from_motion_data(self.get_bvh_saver().motion, ptu.device, torch.float32)
        self.joint_tpose = ptu.from_numpy(self.sim_character.joint_tpose_with_root_joint.reshape(-1, 3))
        self.body_com_tpose = ptu.from_numpy(self.sim_character.body_com_tpose.reshape(-1, 3))
        if self.show:
            self.drawstuff_render_mode = kargs['drawstuff_render_mode']
            self.init_viewer()


    def init_viewer(self):
        if self.show:
            if self.mode == 'drawstuff':
                '''
                use default viewer
                '''
                try:
                    from VclSimuBackend.Render import Renderer
                except ImportError:
                    import VclSimuBackend
                    Renderer = VclSimuBackend.Render
                self.renderObj = Renderer.RenderWorld(self.scene.world)
                viewer_start(self.renderObj, self.drawstuff_render_mode, self.sim_character)
                self.renderObj.track_body(self.sim_character.bodies[0], False)  
                print("rendering...")
    
    @property
    def stastics(self):
        return self.motion_data.stastics
    
    @property
    def sim_character(self):
        return self.scene.characters[0]

    @property
    def ref_character(self):
        return self.scene.characters[1]
    
    def get_bvh_saver(self):
        bvh_saver = VclSimuBackend.ODESim.CharacterTOBVH(self.sim_character, self.fps)
        bvh_saver.bvh_hierarchy_no_root()
        return bvh_saver

    @staticmethod
    def seed( seed):
        SetInitSeed(seed)
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--env_contact_type", type=int, default=0, help="contact type, 0 for LCP and 1 for maxforce")
        arg_parser.add_argument("--env_min_length", type=int, default=26, help="episode won't terminate if length is less than this")
        arg_parser.add_argument("--env_max_length", type=int, default=512, help="episode will terminate if length reach this")
        arg_parser.add_argument("--env_err_threshod", type = float, default = 0.5, help="height error threshod between simulated and tracking character")
        arg_parser.add_argument("--env_err_length", type = int, default = 20, help="episode will terminate if error accumulated ")
        arg_parser.add_argument("--env_scene_fname", type = str, default = "./Data/Misc/muscle_character.json", help="pickle file for scene")
        arg_parser.add_argument("--env_no_balance", default = False, help="whether to use distribution balance when choose initial state", action = 'store_true')
        arg_parser.add_argument("--env_use_com_height", default = False, help="if true, calculate com in terminate condition, else use head height", action = 'store_true')
        arg_parser.add_argument("--env_recompute_velocity", default = True, help = "whether to resample velocity")
        arg_parser.add_argument("--env_random_count", type=int, default=96, help = "target will be random switched for every 96 frame")
        arg_parser.add_argument("--use_ref_motion_set", type=int, default=0, help = "whether use ref motion set")
        arg_parser.add_argument("--motion_dataset", type = str, default = None, help="path to motion dataset")
        
        return arg_parser
    
    def update_and_get_target(self):
        self.step_cur_frame()
        return 
    
    @staticmethod
    def isnotfinite(arr):
        res = np.isfinite(arr)
        return not np.all(res)
    
    def cal_done(self, state, obs):
        height = state[...,self.head_idx,1]
        """
        actually in calc done func
        there maybe no need for resample ... 
        """
        
        reference_state = self.motion_data.state[self.counter]
        target_height = reference_state[self.head_idx,1]
        if abs(height - target_height) > self.err_threshod:
            self.done_cnt +=1
        else:
            self.done_cnt = max(0, self.done_cnt - 1)
        
        if self.isnotfinite(state):
            return 2
        
        if np.any( np.abs(obs) > 50):
            return 2
        
        if self.step_cnt >= self.min_length:
            if self.done_cnt >= self.err_length:
                return 2
            if self.step_cnt >= self.max_length:
                return 1
        return 0
    
    def update_val(self, done, rwd, frame_num):
        tmp_val = self.val / 2
        last_i = 0
        for i in range(frame_num.shape[0]):
            if done[i] !=0:
                tmp_val[frame_num[i]] = rwd[i] if done[i] == 1 else 0
                for j in reversed(range(last_i, i)):
                    tmp_val[frame_num[j]] = 0.95*tmp_val[frame_num[j+1]] + rwd[j]
                last_i = i
        self.val = 0.9 * self.val + 0.1 * tmp_val
        return
        
    def update_p(self):
        self.p = 1/ self.val.clip(min = 0.01)
        self.p_init = self.p[self.init_index]
        self.p_init /= (np.sum(self.p_init) + 1e-6)
        
    def get_info(self):
        int_cnt = int(self.counter)
        return {
            'frame_num': int_cnt
        }
    
    def get_target(self):
        return self.motion_data.observation[self.counter]

    def get_target_scaled_muscle_len(self):
        return self.motion_data.scaled_muscle_len[self.counter]
        
    
    @staticmethod
    def load_character_state(character, state):
        character.load(state_to_BodyInfoState(state))
        aabb = character.get_aabb() 
        if aabb[2]<0:
            character.move_character_by_delta(np.array([0,-aabb[2]+1e-4,0]))
        character.update_anchor_pos()

    def load_character_state_and_last_frame_state(self, state, last_frame_state):
        last_sim_step_BodyInfoState = resample_BodyInfoState(self.env_substep, self.env_fps,  last_frame_state, state)
        self.sim_character.load(last_sim_step_BodyInfoState)
        aabb = self.sim_character.get_aabb() 
        if aabb[2]<0:
            self.sim_character.move_character_by_delta(np.array([0,-aabb[2]+1e-4,0]))
        _, last_frame_muscle_len = self.sim_character.update_anchor_pos()
        self.sim_character.set_last_frame_muscle_len(last_frame_muscle_len)
        self.sim_character.load(state_to_BodyInfoState(state))
        aabb = self.sim_character.get_aabb() 
        if aabb[2]<0:
            self.sim_character.move_character_by_delta(np.array([0,-aabb[2]+1e-4,0]))
        self.sim_character.update_anchor_pos()

    
    def step_counter(self, random = False):
        self.counter += 1
        if random:
            try:  
                self.counter = index_counter.random_select(self.init_index, p = self.p_init)
            except:
                self.counter = index_counter.random_select(self.init_index)

        
    def reset(self, frame = -1, set_state = True ):
        """reset enviroment

        Args:
            frame (int, optional): if not -1, reset character to this frame, target to next frame. Defaults to -1.
            set_state (bool, optional): if false, enviroment will only reset target, not character. Defaults to True.
        """
        self.counter = frame
        if frame == -1:
            self.step_counter(random=True)
        self.begin_counter = self.counter
        if set_state:
            self.load_character_state_and_last_frame_state(self.motion_data.state[self.counter], self.motion_data.state[self.counter-1])
            self.load_character_state(self.ref_character, self.motion_data.state[self.counter])
        self.state = character_state(self.sim_character)
        
        self.observation = state2ob(torch.from_numpy(self.state)).numpy()
        self.observation_rigid = self.observation
        self.muscle_state = np.zeros(15, dtype=np.float64)

        if self.no_fatigue_random_start:
            pass
        else:
            self.sim_character.fatigue_state_random_start(mode_idx=0)
            
        if self.use_muscle_state:
            self.muscle_state = character_muscle_state(self.sim_character)
            self.observation = np.concatenate((self.observation.flatten(), self.muscle_state.flatten()))
        info = self.get_info()
        self.step_counter(random = False)

        self.step_cnt = 0
        self.done_cnt = 0
        output_dict = {
            'state': self.state,
            'muscle_state': self.muscle_state ,
            'observation_rigid': self.observation_rigid,
            'observation': self.observation,
            'target': self.get_target(),
            'target_scaled_muscle_len': self.get_target_scaled_muscle_len()
        }

        modified_ref_state = self.motion_data.state[self.counter]
        modified_body_pos = self.sim_char_root_body_com[ [0,2]] - modified_ref_state[0, [0,2]] +[-1.5,  0.0]
        modified_ref_state[:, [0,2]] = modified_ref_state[:, [0,2]] + modified_body_pos

        self.load_character_state(self.ref_character, modified_ref_state)
        return output_dict, info
    
 
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

        modified_ref_state = self.motion_data.state[self.counter]
        modified_body_pos = self.sim_char_root_body_com[ [0,2]] - modified_ref_state[0, [0,2]] +[-1.5,  0.0]
        modified_ref_state[:, [0,2]] = modified_ref_state[:, [0,2]] + modified_body_pos

        self.load_character_state(self.ref_character, modified_ref_state)
        self.target = self.get_target()
        self.target_scaled_muscle_len = self.get_target_scaled_muscle_len()
        self.step_cnt += 1

    def after_substep(self):
        pass

    def post_step_draw(self, muscle_force, muscle_len):
        activation = self.sim_character.activation_levels 
        activation = activation.astype(np.float32)
        if self.show:
            if self.mode == 'drawstuff':
                if self.drawstuff_render_mode == 0 or self.drawstuff_render_mode == 1:
                    self.renderObj.assign_muscle_property_with_ref(activation, self.sim_character.anchor_positions.reshape(-1), self.ref_character.anchor_positions.reshape(-1))
                else:
                    raise NotImplementedError


    def step_core(self, action, using_yield = False, **kargs):
        ref_muscle_len  = self.get_ref_muscle_length(action).reshape(-1)

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
            self.after_substep()

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
    
    def step(self, action, **kargs):
        step_generator = self.step_core(action, **kargs)
        return next(step_generator)
    
    ## Cpp Functions
    def apply_pd_muscle_force(self, sim_muscle_len, ref_muscle_len=None):
        return self.sim_character.apply_pd_muscle_force(sim_muscle_len, ref_muscle_len)
    
    def apply_fatigue(self):
        return self.sim_character.fatigue_step(self.sim_character.activation_levels)
        
    def get_sim_muscle_length(self):
        _, sim_muscle_len = self.sim_character.update_anchor_pos()
        return sim_muscle_len
    
    def get_ref_muscle_length(self, action):
        ref_char_muscle_len = self.sim_character.muscle_origin_length_calculated.reshape(-1) * (1.0 + action.astype(np.float64) )
        return ref_char_muscle_len
    
    def calc_hill_type_activation(self, muscle_amp, sim_muscle_len):
        activation, unclip_activation = self.sim_character.calc_hill_type_activation(muscle_amp, sim_muscle_len)
        return activation, unclip_activation
    
    def rigid_body_simulate(self):
        self.scene.damped_simulate(1)
    