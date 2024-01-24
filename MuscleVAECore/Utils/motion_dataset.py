from motion_utils import *
from misc import add_to_list
import pytorch_utils as ptu

import VclSimuBackend
try:
    from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
    # from VclSimuBackend.ODESim.TargetPose import TargetPose
    # from VclSimuBackend.ODESim.ODECharacter import ODECharacter
    from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
    from VclSimuBackend.ODESim import CharacterToBVH
    from VclSimuBackend.pymotionlib import BVHLoader
except ImportError:
    BVHToTargetBase = VclSimuBackend.ODESim.BVHToTarget.BVHToTargetBase
    SetTargetToCharacter = VclSimuBackend.ODESim.TargetPose.SetTargetToCharacter
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    BVHLoader = VclSimuBackend.pymotionlib.BVHLoader

class MotionDataSet():
    def __init__(self, fps) -> None:
        """ We orgnaize motion capture data into pickle files

        Args:
            fps (int): target fps of downsampled bvh
        """        
        self.state = None
        self.observation = None
        self.done = None
        self.scaled_muscle_len = None
        self.num_frames = 0
        self.fps = fps
    
    @property
    def stastics(self):
        obs_mean = np.mean(self.observation, axis=0)
        obs_std = np.std(self.observation, axis =0)
        obs_std[obs_std < 1e-1] = 0.1
        
        delta = self.observation[1:] - self.observation[:-1]
        _,_,vel, avel,_,_= decompose_obs(delta)
        num = delta.shape[0]
        delta = np.concatenate([vel.reshape(num,-1,3),avel.reshape(num,-1,3)], axis = -1)
        delta = delta.reshape(num,-1)
        delta_mean = np.mean(delta, axis = 0)
        delta_std = np.std(delta, axis = 0)
        delta_std[delta_std < 1e-1] = 0.1
        return {
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'delta_mean': delta_mean,
            'delta_std': delta_std
        }
    
    def add_bvh_with_character(self, name, character, flip = False):
        if flip:
            target = BVHToTargetBase(name, self.fps, character, flip = np.array([1,0,0])).init_target()
        else:
            target = BVHToTargetBase(name, self.fps, character).init_target()
        tarset : SetTargetToCharacter = SetTargetToCharacter(character, target)

        state, ob, done = [],[],[] 
        
        for i in range(target.num_frames):
            tarset.set_character_byframe(i)
            state_tmp = character_state(character)
            ob_tmp =  state2ob(torch.from_numpy(state_tmp)).numpy()
            done_tmp = (i == (target.num_frames -1))
            state.append(state_tmp[None,...])
            ob.append(ob_tmp.flatten()[None,...])
            done.append(np.array(done_tmp).reshape(1,1))

        
        self.num_frames += target.num_frames 
        self.state = add_to_list(state, self.state)
        self.observation = add_to_list(ob, self.observation)
        self.done = add_to_list(done, self.done)

    def add_bvh_with_character_with_scaled_muscle_len(self, name, character, flip = False):
        if flip:
            target = BVHToTargetBase(name, self.fps, character, flip = np.array([1,0,0])).init_target()
        else:
            target = BVHToTargetBase(name, self.fps, character).init_target()
        tarset : SetTargetToCharacter = SetTargetToCharacter(character, target)

        state, ob, done, scaled_muscle_len = [],[],[], []
        
        for i in range(target.num_frames):
            tarset.set_character_byframe(i)
            state_tmp = character_state(character)
            ob_tmp =  state2ob(torch.from_numpy(state_tmp)).numpy()
            len_tmp = np_state2scaled_muscle_length(state_tmp, character)
            done_tmp = (i == (target.num_frames -1))
            state.append(state_tmp[None,...])
            ob.append(ob_tmp.flatten()[None,...])
            done.append(np.array(done_tmp).reshape(1,1))
            scaled_muscle_len.append(len_tmp[None,...])

        self.num_frames += target.num_frames 
        self.state = add_to_list(state, self.state)
        self.observation = add_to_list(ob, self.observation)
        self.done = add_to_list(done, self.done)
        self.scaled_muscle_len = add_to_list(scaled_muscle_len, self.scaled_muscle_len)

         
    def add_folder_bvh(self, name, character, mirror_augment = True):
        """Add every bvh in a forlder into motion dataset

        Args:
            name (str): path of bvh folder
            character (ODECharacter): the character of ode
            mirror_augment (bool, optional): whether to use mirror augment. Defaults to True.
        """                
        for file in os.listdir(name):
            if '.bvh' in file:
                print(f'add {file}')
                self.add_bvh_with_character(os.path.join(name, file), character)
        if mirror_augment:
            for file in os.listdir(name):
                if '.bvh' in file:
                    print(f'add {file} flip')
                    self.add_bvh_with_character(os.path.join(name, file), character, flip = True)

    def add_folder_bvh_with_scaled_muscle_len(self, name, character, mirror_augment = True):
        """Add every bvh in a forlder into motion dataset

        Args:
            name (str): path of bvh folder
            character (ODECharacter): the character of ode
            mirror_augment (bool, optional): whether to use mirror augment. Defaults to True.
        """                
        for file in os.listdir(name):
            if '.bvh' in file:
                print(f'add {file}')
                self.add_bvh_with_character_with_scaled_muscle_len(os.path.join(name, file), character)
        if mirror_augment:
            for file in os.listdir(name):
                if '.bvh' in file:
                    print(f'add {file} flip')
                    self.add_bvh_with_character_with_scaled_muscle_len(os.path.join(name, file), character, flip = True)
    
    
     