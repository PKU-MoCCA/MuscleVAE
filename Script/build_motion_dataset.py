import yaml
import argparse
import pickle
from Muscle.muscle_scene_loader import JsonSceneWithMuscleLoader
from MuscleVAECore.Utils.motion_dataset import MotionDataSet
from MuscleVAECore.Utils.misc import *


if __name__ == '__main__':
    '''
    convert mocap bvh into binary file, the bvh will be downsampled and some 
    important data(such as state and observation of each frame) 
    '''    
    parser = argparse.ArgumentParser()
    parser.add_argument("--using_yaml",  default=True, help="if true, configuration will be specified with a yaml file", action='store_true')
    parser.add_argument("--bvh_folder", type=str, default="", help="name of reference bvh folder")
    parser.add_argument("--env_fps", type=int, default=20, help="target FPS of downsampled reference motion")
    parser.add_argument("--env_scene_fname", type = str, default = "odecharacter_scene.pickle", help="pickle file for scene")
    parser.add_argument("--motion_dataset", type=str, default=None, help="name of output motion dataset")
    parser.add_argument("--whether_flip", type=str, default=True, help="whether add fliped bvh")
    args = vars(parser.parse_args())
    
    if args['using_yaml']:
        config = load_yaml(initialdir='Data/Parameters/')
        args.update(config) 
        
    
    scene_loader = JsonSceneWithMuscleLoader()
    scene = scene_loader.file_load(args['env_scene_fname'])
    motion = MotionDataSet(args['env_fps'])
    
    assert args['bvh_folder'] is not None
    motion.add_folder_bvh_with_scaled_muscle_len(args['bvh_folder'], scene.character0, mirror_augment =args['whether_flip'])
    f = open(args['motion_dataset'], 'wb')
    pickle.dump(motion, f) 