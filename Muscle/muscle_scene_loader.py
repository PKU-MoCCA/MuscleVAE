import numpy as np

import VclSimuBackend
from typing import Dict, Any, List

try:
    from VclSimuBackend.ODESim import JsonSceneLoader as JsonSceneLoader
    from VclSimuBackend.ODESim import JsonCharacterLoader as JsonCharacterLoader
except ImportError:
    JsonSceneLoader = VclSimuBackend.ODESim.JsonSceneLoader
    JsonCharacterLoader = VclSimuBackend.ODESim.JsonCharacterLoader
    
    
from Muscle.muscle_character import ODEMuscleCharacter, ODEMuscleCharacterInit

class JsonSceneWithMuscleLoader(JsonSceneLoader):
    def __init__(self, scene=None, is_running: bool = False, muscle_file=''):
        super().__init__(scene, is_running)
        self.muscle_file = muscle_file

    def load_character_list(self, mess_dict: Dict[str, Any]):
        character_info_list: List[Dict[str, Any]] = mess_dict["Characters"]
        for character_info in character_info_list:
            loader = JsonMuscleCharacterLoader(self.world, self.space, self.scene,self.use_hinge, self.use_angle_limit, muscle_file=self.muscle_file)
            character = loader.load(character_info)
            self.characters.append(character)

        return self.characters

class JsonMuscleCharacterLoader(JsonCharacterLoader):
    def __init__(
        self,
        world,
        space,
        scene,
        use_hinge: bool = True,
        use_angle_limit: bool = True,
        ignore_parent_collision: bool = True,
        ignore_grandpa_collision: bool = True,
        load_scale: float = 1.0,
        use_as_base_class: bool = False,
        muscle_file : str = '' 
    ):
        """
        Our character model is defined at world coordinate.
        """
        self.world = world
        self.space = space
        self.scene = scene
        self.use_hinge = use_hinge
        self.use_angle_limit = use_angle_limit
        self.muscle_file = muscle_file

        self.ignore_parent_collision: bool = ignore_parent_collision
        self.ignore_grandpa_collision: bool = ignore_grandpa_collision
        self.load_scale: float = load_scale
        self.ignore_list: List[List[int]] = []

        if not use_as_base_class:
            self.character = ODEMuscleCharacter(world, space)
            self.character_init = ODEMuscleCharacterInit(self.character)
        else:
            self.character = None
            self.character_init = None

        self.geom_type = VclSimuBackend.GeomType()
        self.default_friction: float = 0.8 


    def muscle_to_body_map(self, input_body:str):
        MASS_to_stdhuman_dict_full_flip = {
            "Pelvis": "pelvis",
            "Spine": "lowerback",
            "Torso": "torso",
            "FemurR": "lUpperLeg",
            "FemurL": "rUpperLeg",
            "TibiaR": "lLowerLeg",
            "TibiaL": "rLowerLeg",
            "TalusR": "lFoot",
            "TalusL": "rFoot",
            "FootThumbR": "FootThumbL",
            "FootThumbL": "FootThumbR",
            "Head": "Head",
            "ShoulderR": "lClavicle",
            "ShoulderL": "rClavicle",
            "ArmR": "lUpperArm",
            "ArmL": "rUpperArm",
            "ForeArmR": "lLowerArm",
            "ForeArmL": "rLowerArm",
            "HandR": "lHand",
            "HandL": "rHand",
            "Neck": "Neck",
            "FootPinkyR": "FootPinkyL",
            "FootPinkyL": "FootPinkyR"
        }

        return MASS_to_stdhuman_dict_full_flip[input_body]

    def str_to_numpy(self, input_str: list):
        return np.array([np.float64(input_str[i]) for i in range(len(input_str))])

    def find_bodyId_by_name(self, body_name: str):
        idx = 0
        for body in self.character.bodies:
            if body.name == body_name:
                return idx
            idx += 1
        return None

    def findGropuIdByName(self, group_name):
        idx = 0
        for each_group_name in self.character.muscle_group_names:
            if each_group_name == group_name:
                return idx
            idx += 1
        return None

    def getPosRelPointNumpyAccordingGeoms0Transform(self, body, global_q):
        rot = body.geoms[0].RotationNumpy.reshape(3, 3)
        translation = body.geoms[0].PositionNumpy
        return np.matmul(rot.transpose() , (global_q - translation))

    def getGeoms0Transform(self, body):
        rot = body.geoms[0].RotationNumpy
        return rot

    def getGeom0OffsetRotation(self, body):
        rot = body.geoms[0].getOffsetRotationNumpy()
        return rot

    def load_muscles(self, muscle_parts: List):
        """
        Load muscles in json
        """
        self.character.sim_hz = self.scene.sim_fps
        self.character.bodies_num = len(self.bodies)
        self.character.geoms0_gid = self.character.body_info.get_geoms0_gids()
        muscle_parts.sort(key=lambda x: x["muscleID"])  
        self.character.muscle_num = len(muscle_parts)
        for muscle_piece in muscle_parts:
            waypoints = muscle_piece.get("Waypoints")
            self.character.anchors_num += len(waypoints)
            group_name = muscle_piece.get("group_name")
            if group_name not in self.character.muscle_group_names:
                self.character.muscle_group_names.append(group_name) 
        self.character.muscle_groups_num = len(self.character.muscle_group_names)
        self.character.muscles_num = len(muscle_parts)

        self.character.activation_levels = np.zeros([self.character.muscles_num], dtype=np.float64)
        self.character.anchor_positions = np.zeros([self.character.anchors_num, 3], dtype=np.float64)
        self.character.anchor_positions_medium0 = np.zeros([self.character.anchors_num, 3], dtype=np.float64)
        self.character.anchor_positions_medium1 = np.zeros([self.character.anchors_num, 3], dtype=np.float64)

        self.character.anchor_local_postions = np.zeros([self.character.anchors_num, 3, 2], dtype=np.float64)
        self.character.anchor_body0_bid = np.zeros([self.character.anchors_num], dtype=np.uint64)
        self.character.anchor_body1_bid = np.zeros([self.character.anchors_num], dtype=np.uint64)
        self.character.anchor_body0_geom0_gid = np.zeros([self.character.anchors_num], dtype=np.uint64)
        self.character.anchor_body1_geom0_gid = np.zeros([self.character.anchors_num], dtype=np.uint64)
        self.character.anchor_body0_idx = np.zeros([self.character.anchors_num], dtype=np.int32)
        self.character.anchor_body1_idx = np.zeros([self.character.anchors_num], dtype=np.int32)

        self.character.muscle_properties = np.zeros([self.character.muscles_num , 5], dtype=np.float64)
        self.character.mask_anchor_to_body = np.zeros([self.character.bodies_num, self.character.anchors_num], dtype=np.int32)
        
        self.character.muscle_in_which_group = np.zeros([self.character.muscles_num], dtype=np.int32)
        self.character.group_hold_muscle_num = np.zeros([self.character.muscle_groups_num], dtype=np.int32)
        self.character.muscle_hold_anchor_num = np.zeros([self.character.muscles_num], dtype=np.int32)
        self.character.mask_anchor_to_muscle = np.zeros([self.character.muscles_num , self.character.anchors_num], dtype=np.int32)
        self.character.mask_anchor_to_muscle_without_padding = np.zeros([self.character.muscles_num , self.character.anchors_num], dtype=np.int32)
        self.character.mask_anchor_for_force_direction = np.zeros([self.character.anchors_num], dtype=np.int32)
        self.character.reversed_mask_anchor_for_force_direction = np.zeros([self.character.anchors_num], dtype=np.int32)
        self.character.map_anchor_to_act_force_body = np.zeros([self.character.anchors_num], dtype=np.int32)
        self.character.map_anchor_to_act_force_body_cid = np.zeros([self.character.anchors_num], dtype=np.uint64)

        self.character.mask_weights_anchor_to_body = np.zeros([self.character.bodies_num, self.character.anchors_num], dtype=np.float64)

        self.character.mask_anchor_to_body0_weight = np.zeros([self.character.bodies_num, self.character.anchors_num], dtype=np.float64)
        self.character.mask_anchor_to_body1_weight = np.zeros([self.character.bodies_num, self.character.anchors_num], dtype=np.float64)
        self.character.mask_anchor_to_body0_index = np.zeros([self.character.bodies_num, self.character.anchors_num], dtype=np.float64)
        self.character.mask_anchor_to_body1_index = np.zeros([self.character.bodies_num, self.character.anchors_num], dtype=np.float64)

        self.character.anchor_act_force_on_which_body = np.zeros([self.character.anchors_num], dtype=np.int32)
        self.character.anchor_act_force_on_which_body_second = np.zeros([self.character.anchors_num], dtype=np.int32)
        self.character.anchor_act_force_on_which_body_name = []
        self.character.anchor_act_force_on_which_body_second_name = []
        self.character.anchor_act_force_on_which_body_two = np.zeros([self.character.anchors_num, 2], dtype=np.int32)
        self.character.anchor_act_force_weights = np.ones([self.character.anchors_num], dtype=np.float64)

        self.character.muscle_part_idx :np.ndarray = np.zeros([self.character.muscles_num], dtype=np.int32)
        self.character.muscle_activated_proportion :np.ndarray = np.zeros([self.character.muscles_num], dtype=np.float64)
        self.character.muscle_resting_proportion :np.ndarray = np.ones([self.character.muscles_num], dtype=np.float64)
        self.character.muscle_fatigued_proportion :np.ndarray = np.zeros([self.character.muscles_num], dtype=np.float64)
        self.character.muscle_target_load :np.ndarray = np.zeros([self.character.muscles_num], dtype=np.float64)
        self.character.part_fatigue_weight: np.ndarray = np.zeros([self.character.muscles_num], dtype=np.float64)
        self.character.part_activated_proportion :np.ndarray = np.zeros([5], dtype=np.float64)
        self.character.part_resting_proportion :np.ndarray = np.ones([5], dtype=np.float64)
        self.character.part_fatigued_proportion :np.ndarray = np.zeros([5], dtype=np.float64)
        self.pose_aware_f0_max = np.zeros([self.character.muscles_num], dtype=np.float64)
        self.character.group_avg_coact_matrix = np.zeros((self.character.muscles_num * self.character.muscles_num), dtype=np.float64)
        self.character.geom_toward_body_rot_matrix = np.zeros((self.character.bodies_num, 9), dtype=np.float64)

        ## for json arrange
        self.character.muscle_origin_body_name = []
        self.character.muscle_insertion_body_name = []
        self.character.muscle_in_which_group_name = []
        
        muscle_id = 0
        anchor_id = 0		 
        for muscle_piece in muscle_parts:
            name = muscle_piece.get("Name")
            group_name = muscle_piece.get("group_name")
            try:
                f0 = np.float64(muscle_piece.get("f0"))
            except:
                f0 = np.float64(muscle_piece.get("f0_MASS"))
            lm = np.float64(muscle_piece.get("lm"))
            lt = np.float64(muscle_piece.get("lt"))
            pen_angle = np.float64(muscle_piece.get("pen_angle"))
            lmax = -np.float64(muscle_piece.get("lmax"))
            self.character.muscle_names.append(name)
            self.character.muscle_properties[muscle_id] = np.array([f0, lm, lt, pen_angle, lmax])
            waypoints = muscle_piece.get("Waypoints")
            len_waypoints = len(waypoints)
            self.character.muscle_hold_anchor_num[muscle_id] = len_waypoints
            self.character.muscle_in_which_group[muscle_id] = self.findGropuIdByName(group_name)
            self.character.muscle_in_which_group_name.append(group_name)
            self.character.group_hold_muscle_num[self.findGropuIdByName(group_name)] += 1

            # for json arrange
            for i in range(len_waypoints):
                _body = waypoints[i].get("BodyName")
                if i == 0:
                    self.character.muscle_origin_body_name.append(_body)
                if i == len_waypoints - 1:
                    self.character.muscle_insertion_body_name.append(_body)
            
        
            for i in range(len_waypoints):
                _body = waypoints[i].get("BodyName")
                self.character.anchor_origin_body_name.append(_body)
                _body_name = self.muscle_to_body_map(_body)
                _p = waypoints[i].get("Position")
                _p_numpy = self.str_to_numpy(_p) 
                _p_numpy[0] = - _p_numpy[0]
                body_id = self.find_bodyId_by_name(_body_name)
                self.character.mask_anchor_to_muscle_without_padding[muscle_id, anchor_id + i] = 1
                if i < len_waypoints -1 :
                    self.character.mask_anchor_to_muscle[muscle_id, anchor_id + i] = 1
                    self.character.mask_anchor_for_force_direction[anchor_id + i] = 1
                self.character.anchor_positions[anchor_id + i] = _p_numpy
                

                if i == 0 or i == len_waypoints - 1:
                    if i == 0 :
                        self.character.muscle_origin_body_idx.append(body_id)
                    else:
                        self.character.muscle_insertion_body_idx.append(body_id)
                        if _body in ['FemurR', 'TibiaR', 'TalusR', 'FootPinkyR', 'FootThumbR']:
                            self.character.muscle_part_idx[muscle_id] = 1
                        elif _body in ['FemurL', 'TibiaL', 'TalusL', 'FootPinkyL', 'FootThumbL']:
                            self.character.muscle_part_idx[muscle_id] = 2
                        elif _body in ['ShoulderR', 'ArmR', 'ForeArmR', 'HandR']:
                            self.character.muscle_part_idx[muscle_id] = 3
                        elif _body in ['ShoulderL', 'ArmL', 'ForeArmL', 'HandL']:
                            self.character.muscle_part_idx[muscle_id] = 4

                    self.character.mask_weights_anchor_to_body[body_id, anchor_id + i] = 1
                    
                    self.character.mask_anchor_to_body0_weight[body_id, anchor_id + i] = 1.0
                    self.character.mask_anchor_to_body0_index[body_id, anchor_id + i] = 1.0
                    self.character.map_anchor_to_act_force_body[anchor_id + i] = body_id
                    self.character.map_anchor_to_act_force_body_cid[anchor_id + i] = self.character.bodies[body_id].get_bid()
                    self.character.anchor_local_postions[anchor_id + i, :, 0] = self.getPosRelPointNumpyAccordingGeoms0Transform(self.character.bodies[body_id], _p_numpy)
                    self.character.anchor_body0_bid[anchor_id + i] = self.character.bodies[body_id].get_bid()
                    self.character.anchor_body0_geom0_gid[anchor_id + i] = self.character.geoms0_gid[body_id]
                    self.character.anchor_body0_idx[anchor_id + i] = body_id
                    
                    self.character.anchor_act_force_on_which_body_two[anchor_id + i, 0] = body_id
                    self.character.anchor_act_force_on_which_body_two[anchor_id + i, 1] = -1
                    self.character.anchor_act_force_on_which_body[anchor_id + i] = body_id
                    self.character.anchor_act_force_on_which_body_second[anchor_id + i] = -1
                    self.character.anchor_act_force_on_which_body_name.append(self.character.bodies[body_id].name) 
                    self.character.anchor_act_force_on_which_body_second_name.append(None) 
                else:
                    self.character.displacement_norm_anchor_in_body = np.zeros(self.character.bodies_num)
                    self.character.displacement_norm_anchor_in_body[0] = np.linalg.norm(self.character.bodies[0].PositionNumpy - _p_numpy)
                    for j in range(1, self.character.bodies_num):
                        self.character.displacement_norm_anchor_in_body[j] = np.linalg.norm((self.character.joints[self.character.child_body_to_joint[j]].getAnchorNumpy() + self.character.joints[self.character.child_body_to_joint[j]].getAnchor2Numpy())/2 - _p_numpy)
                    
                    self.character.displacement_norm_anchor_in_body_cp.append(self.character.displacement_norm_anchor_in_body)
                    self.character.displacement_norm_anchor_in_body_cp_sorted.append(np.sort(self.character.displacement_norm_anchor_in_body))
                    self.character.displacement_norm_anchor_in_body_cp_index[anchor_id + i]= len(self.character.displacement_norm_anchor_in_body_cp_sorted)-1


                    nearest_body_id = np.argmin(self.character.displacement_norm_anchor_in_body) 
                    if self.character.displacement_norm_anchor_in_body[nearest_body_id ] < 0.08:
                        if self.character.body_info.parent[nearest_body_id] == -1:
                            self.character.mask_weights_anchor_to_body[nearest_body_id, anchor_id + i] = 1
                            self.character.mask_anchor_to_body[nearest_body_id, anchor_id + i] = 1
                            
                            self.character.anchor_local_postions[anchor_id + i, :, 0] = self.getPosRelPointNumpyAccordingGeoms0Transform(self.character.bodies[nearest_body_id], _p_numpy)
                            self.character.anchor_body0_bid[anchor_id + i] = self.character.bodies[nearest_body_id].get_bid()
                            self.character.mask_anchor_to_body0_weight[nearest_body_id, anchor_id + i] = 1.0
                            self.character.mask_anchor_to_body0_index[nearest_body_id, anchor_id + i] = 1.0
                            self.character.map_anchor_to_act_force_body[anchor_id + i] = nearest_body_id
                            self.character.map_anchor_to_act_force_body_cid[anchor_id + i] = self.character.bodies[nearest_body_id].get_bid()
                            self.character.anchor_body0_geom0_gid[anchor_id + i] = self.character.geoms0_gid[nearest_body_id]
                            self.character.anchor_body0_idx[anchor_id + i] = nearest_body_id

                            self.character.anchor_act_force_on_which_body_two[anchor_id + i, 0] = nearest_body_id
                            self.character.anchor_act_force_on_which_body_two[anchor_id + i, 1] = -1
                            self.character.anchor_act_force_on_which_body[anchor_id + i] = nearest_body_id
                            self.character.anchor_act_force_on_which_body_second[anchor_id + i] = -1
                            self.character.anchor_act_force_on_which_body_name.append(self.character.bodies[nearest_body_id].name) 
                            self.character.anchor_act_force_on_which_body_second_name.append(None) 
                        else:	
                            self.character.two_bodies_anchor_num += 1
                            self.character.two_bodies_anchor_list.append(anchor_id +i)

                            total_weight = 0.0					
                            body0_weight = 1.0 / (np.sqrt(self.character.displacement_norm_anchor_in_body[nearest_body_id]))		
                            total_weight += body0_weight
                            parent_of_nearest = self.character.body_info.parent[nearest_body_id]
                            body1_weight = 1.0 / (np.sqrt(self.character.displacement_norm_anchor_in_body[parent_of_nearest]))
                            total_weight += body1_weight
                            body0_weight /= total_weight
                            body1_weight /= total_weight
                            self.character.mask_weights_anchor_to_body[nearest_body_id, anchor_id + i] = body0_weight
                            self.character.mask_weights_anchor_to_body[parent_of_nearest, anchor_id + i] = body1_weight
                            self.character.mask_anchor_to_body[nearest_body_id, anchor_id + i] = 1
                            
                            self.character.map_anchor_to_act_force_body[anchor_id + i] = nearest_body_id
                            self.character.map_anchor_to_act_force_body_cid[anchor_id + i] = self.character.bodies[nearest_body_id].get_bid()
                            self.character.anchor_local_postions[anchor_id + i, :, 0] = self.getPosRelPointNumpyAccordingGeoms0Transform(self.character.bodies[nearest_body_id], _p_numpy)
                            self.character.anchor_local_postions[anchor_id + i, :, 1] = self.getPosRelPointNumpyAccordingGeoms0Transform(self.character.bodies[parent_of_nearest], _p_numpy)
                            self.character.anchor_body0_bid[anchor_id + i] = self.character.bodies[nearest_body_id].get_bid()
                            self.character.anchor_body1_bid[anchor_id + i] = self.character.bodies[parent_of_nearest].get_bid()
                            self.character.anchor_body0_geom0_gid[anchor_id + i] = self.character.geoms0_gid[nearest_body_id]
                            self.character.anchor_body1_geom0_gid[anchor_id + i] = self.character.geoms0_gid[parent_of_nearest]
                            self.character.anchor_body0_idx[anchor_id + i] = nearest_body_id
                            self.character.anchor_body1_idx[anchor_id + i] = parent_of_nearest
                            
                            self.character.mask_anchor_to_body0_weight[nearest_body_id, anchor_id + i] = body0_weight
                            self.character.mask_anchor_to_body1_weight[parent_of_nearest, anchor_id + i] = body1_weight
                            self.character.mask_anchor_to_body0_index[nearest_body_id, anchor_id + i] = 1.0
                            self.character.mask_anchor_to_body1_index[parent_of_nearest, anchor_id + i] = 1.0

                            self.character.anchor_act_force_on_which_body_two[anchor_id + i, 0] = nearest_body_id
                            self.character.anchor_act_force_on_which_body_two[anchor_id + i, 1] = parent_of_nearest
                            self.character.anchor_act_force_on_which_body[anchor_id + i] = nearest_body_id
                            self.character.anchor_act_force_on_which_body_second[anchor_id + i] = parent_of_nearest
                            self.character.anchor_act_force_on_which_body_second_name.append(self.character.bodies[parent_of_nearest].name) 
                            self.character.anchor_act_force_weights[anchor_id + i] = body0_weight
                            self.character.anchor_act_force_on_which_body_name.append(self.character.bodies[nearest_body_id].name) 

                    else:
                        self.character.mask_weights_anchor_to_body[body_id, anchor_id + i] = 1.0
                        self.character.mask_anchor_to_body[body_id, anchor_id + i] = 1
                        
                        self.character.map_anchor_to_act_force_body[anchor_id + i] = body_id
                        self.character.map_anchor_to_act_force_body_cid[anchor_id + i] = self.character.bodies[body_id].get_bid()
                        self.character.anchor_local_postions[anchor_id + i, :, 0] = self.getPosRelPointNumpyAccordingGeoms0Transform(self.character.bodies[body_id], _p_numpy)
                        self.character.anchor_body0_bid[anchor_id + i] = self.character.bodies[body_id].get_bid()
                        self.character.anchor_body0_geom0_gid[anchor_id + i] = self.character.geoms0_gid[body_id]
                        self.character.anchor_body0_idx[anchor_id + i] = body_id
                        self.character.mask_anchor_to_body0_weight[body_id, anchor_id + i] = 1.0
                        self.character.mask_anchor_to_body0_index[body_id, anchor_id + i] = 1.0
                        
                        self.character.anchor_act_force_on_which_body_two[anchor_id + i, 0] = body_id
                        self.character.anchor_act_force_on_which_body_two[anchor_id + i, 1] = -1
                        self.character.anchor_act_force_on_which_body[anchor_id + i] = body_id
                        self.character.anchor_act_force_on_which_body_second[anchor_id + i] = -1
                        self.character.anchor_act_force_on_which_body_second_name.append(None) 
                            
                        self.character.anchor_act_force_on_which_body_name.append(self.character.bodies[body_id].name) 

            muscle_id += 1
            anchor_id += len_waypoints
        self.character.reversed_mask_anchor_for_force_direction[1:] = self.character.mask_anchor_for_force_direction[:-1]
        self.character.body_parent_index = np.array(self.character.body_info.parent, dtype=np.int32)
        self.character.body_com_tpose = np.zeros((self.character.bodies_num, 3), dtype=np.float64)
        self.character.joint_tpose = np.zeros((self.character.bodies_num - 1, 3), dtype=np.float64)
        self.character.joint_tpose_with_root_joint = np.zeros((self.character.bodies_num, 3), dtype=np.float64)
        kp_coff_ratio = 2.0
        self.character.kp_coff = self.character.muscle_properties[:, 0] * float(kp_coff_ratio)
        self.character.pd_muscle_amp = None
        self.character.calc_body_com_and_joint_tpose()
        self.character.calc_tpose_geom0_towards_body0_offset()

        self.character.sparse_anchor_body0_local_pos = np.einsum('ij, ik->ijk', self.character.mask_anchor_to_body0_index.transpose(), self.character.anchor_local_postions[:,:, 0])
        self.character.sparse_anchor_body1_local_pos = np.einsum('ij, ik->ijk', self.character.mask_anchor_to_body1_index.transpose(), self.character.anchor_local_postions[:,:, 1])
        _, self.character.muscle_origin_length_calculated = self.character.update_anchor_pos()

        
    def load(self, mess_dict: Dict[str, Any]):
        """
        Load ODE Character from json file
        """
        self.ignore_parent_collision &= mess_dict.get("IgnoreParentCollision", True)
        self.ignore_grandpa_collision &= mess_dict.get("IgnoreGrandpaCollision", True)

        name = mess_dict.get("CharacterName")
        if name:
            self.character.name = name

        label = mess_dict.get("CharacterLabel")
        if label:
            self.character.label = label

        self.load_bodies(mess_dict["Bodies"])

        joints: List[Dict[str, Any]] = mess_dict.get("Joints")
        if joints:
            self.load_joints(joints)

        self.character_init.init_after_load(mess_dict["CharacterID"],
                                            self.ignore_parent_collision,
                                            self.ignore_grandpa_collision)

        muscles = mess_dict.get("Muscles")
        if muscles:
            self.load_muscles(muscles)

        self_colli = mess_dict.get("SelfCollision")
        if self_colli is not None:
            self.character.self_collision = self_colli

        kinematic = mess_dict.get("Kinematic")
        if kinematic is not None:
            self.character.is_kinematic = kinematic

        pd_param: Dict[str, List[float]] = mess_dict.get("PDControlParam")
        if pd_param:
            self.load_pd_control_param(pd_param)

        end_joints: List[Dict[str, Any]] = mess_dict.get("EndJoints")
        if end_joints:
            self.load_endjoints(end_joints)

        init_root: Dict[str, Any] = mess_dict.get("RootInfo")
        if init_root:
            self.load_init_root_info(init_root)

        return self.character
