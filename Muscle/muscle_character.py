import numpy as np
import torch
import VclSimuBackend
from scipy.spatial.transform import Rotation as R

try:
    from VclSimuBackend.ODESim.ODECharacter import ODECharacter
    from VclSimuBackend.ODESim import ODECharacterInit as ODECharacterInit
except ImportError:
    ODECharacter = VclSimuBackend.ODESim.ODECharacter.ODECharacter
    ODECharacterInit = VclSimuBackend.ODESim.ODECharacterInit

class ODEMuscleCharacterInit(ODECharacterInit):
    def __init__(self, character):
        super().__init__(character)

class ODEMuscleCharacter(ODECharacter):
    def __init__(self, world, space):
        super().__init__(world, space)
        self.fps = 120
        self.sim_nz = 120
        self.anchor_positions: np.ndarray = None
        self.anchor_positions_medium0: np.ndarray = None
        self.anchor_positions_medium1: np.ndarray = None

        self.anchor_local_postions: np.ndarray = None
        self.anchor_body0_bid : np.ndarray = None
        self.anchor_body1_bid : np.ndarray = None
        self.anchor_body0_geom0_gid : np.ndarray = None
        self.anchor_body1_geom0_gid : np.ndarray = None
        self.anchor_body0_idx : np.ndarray = None
        self.anchor_body1_idx : np.ndarray = None
        self.anchor_act_force_on_which_body : np.ndarray = None
        self.anchor_act_force_on_which_body_second : np.ndarray = None
        self.anchor_act_force_on_which_body_name : np.ndarray = None
        self.anchor_act_force_on_which_body_two : np.ndarray = None
        self.anchor_act_force_weights : np.ndarray = None
        self.displacement_norm_anchor_in_body : np.ndarray = None
        self.anchor_origin_body_name = []
        self.muscle_properties: np.ndarray = None
        self.muscle_names = []
        self.muscle_group_names = []
        self.muscle_origin_body_idx = []
        self.muscle_insertion_body_idx = []
        self.muscle_origin_length_calculated: np.ndarray = None
        self.muscle_origin_length_get_flag: bool = False
        self.use_calculated_muscle_origin_length : bool = True
        self.muscle_lengths: np.ndarray = None
        self.unscaled_muscle_lengths: np.ndarray = None
        self.muscle_hold_anchor_num: np.ndarray = None
        self.muscle_in_which_group: np.ndarray = None
        self.group_hold_muscle_num: np.ndarray = None
        self.muscle_origin_length_of_tendon_and_fiber : np.ndarray = None
        self.mask_anchor_to_body0_weight: np.ndarray = None
        self.mask_anchor_to_body1_weight: np.ndarray = None
        self.mask_anchor_to_body0_index: np.ndarray = None
        self.mask_anchor_to_body1_index: np.ndarray = None
        self.mask_anchor_to_body: np.ndarray = None
        self.mask_anchor_to_muscle: np.ndarray = None 					
        self.mask_anchor_to_muscle_without_padding : np.ndarray = None
        self.mask_anchor_for_force_direction: np.ndarray = None
        self.reversed_mask_anchor_for_force_direction : np.ndarray = None
        self.map_anchor_to_act_force_body: np.ndarray = None
        self.map_anchor_to_act_force_body_cid: np.ndarray = None
        self.mask_weights_anchor_to_body: np.ndarray = None
        self.specfic_relative_displacement:np.ndarray = None
        self.activation_levels: np.ndarray = None
        self.child_body_to_joint : np.ndarray = None
        self.muscle_at_body_indices = []
        self.bodies_jacobian_v: np.ndarray = None 
        self.bodies_jacobian_w: np.ndarray = None
        self.anchor_to_body_jacobian_v_transpose : np.ndarray = None
        self.sparse_anchor_body0_local_pos : np.ndarray = None
        self.sparse_anchor_body1_local_pos : np.ndarray = None


        self.displacement_norm_anchor_in_body_cp_sorted = []
        self.displacement_norm_anchor_in_body_cp_index = {}
        self.displacement_norm_anchor_in_body_cp = []
        self.anchors_num = 0
        self.muscles_num = 0
        self.bodies_num = 0 
        self.two_bodies_anchor_num = 0
        self.two_bodies_anchor_list = []
        self.k_pe : np.float64 = 4.0
        self.e_mo : np.float64 = 0.6
        self.gamma : np.float64 = 0.5

        self.muscle_part_idx :np.ndarray = None
        self.muscle_activated_proportion :np.ndarray = None
        self.muscle_resting_proportion :np.ndarray = None
        self.muscle_fatigued_proportion :np.ndarray = None
        self.muscle_target_load :np.ndarray = None
        self.part_fatigue_weight: np.ndarray = None
        self.integral_3cc_model_activated :np.ndarray = None
        self.integral_3cc_model_resting :np.ndarray = None
        self.integral_3cc_model_fatigued :np.ndarray = None
        self.part_activated_proportion :np.ndarray = None
        self.part_resting_proportion :np.ndarray = None
        self.part_fatigued_proportion :np.ndarray = None
        self.r_para = 2.0
        self.F_para = 0.05
        self.R_para = 0.01
        self.LR_para = 50.0
        self.LD_para = 50.0
        self.part_colors = np.array([255, 255, 0, 0, 0, 255, 255, 192, 203, 0, 255, 0, 255, 0, 0 ], dtype=np.float32)/256.0


        self.muscle_maximum_force_up_bound = 5000.0
        self.hill_f0_collect_times = 0
        self.joint_t0_collect_times = 0
        self.pose_aware_f0_max :np.ndarray = None

        self.group_avg_coact_matrix :np.ndarray = None
        self.geom_toward_body_rot_matrix : np.ndarray = None
        self.geoms0_gid : np.ndarray = None
        self.geom0_body0_offset_tpose : np.ndarray = None
        self.muscle_render_mode = 0

    
    @property
    def child_body_to_joint_numpy(self) -> np.ndarray:
        return np.array(self.child_body_to_joint, dtype=np.int32)

    def adjust_muscle_kp_coff(self, kp_input):
        self.kp_coff = kp_input

    def adjust_muscle_kd_coff(self, kp_input):
        self.kd_coff = np.sqrt(kp_input) * 2.0

    def adjust_negative_muscle_kd_coff(self, kp_input):
        self.kd_coff = -np.sqrt(kp_input) * 2.0

    def set_simulate_fps(self, fps_input):
        self.fps = fps_input


    def adjust_joint_kd_coff(self, kd_input):
        self.world.cython_adjust_joint_kd_same(self.joint_info.joint_c_id, kd_input.reshape(-1))

    def adjust_pose_aware_f0_max(self, f0_input):
        self.pose_aware_f0_max = f0_input

    def adjust_joint_t0_max(self, t0_input):
        self.joint_t0_max = t0_input

    def calc_tpose_geom0_towards_body0_offset(self):
        geom0_pos = self.cython_geom_pos().reshape(-1, 3)[0]
        body0_pos = self.body_info.get_body_pos()[0]
        x = geom0_pos[0] - body0_pos[0]
        y = geom0_pos[1] - body0_pos[1]
        z = geom0_pos[2] - body0_pos[2]
        r = np.sqrt(x*x+y*y+z*z)
        self.geom0_body0_offset_tpose = np.array([0.0, r, 0.0])
        return self.geom0_body0_offset_tpose

    def calc_body_com_and_joint_tpose(self):
        for i in range(self.bodies_num):
            self.body_com_tpose[i] = self.bodies[i].getPositionNumpy()
        self.joint_tpose_with_root_joint[0] = self.bodies[0].getPositionNumpy()
        for j in range(1, self.bodies_num):
            self.joint_tpose[j-1] = self.joints[self.child_body_to_joint[j]].getAnchorNumpy() /2 + self.joints[self.child_body_to_joint[j]].getAnchor2Numpy() /2
            self.joint_tpose_with_root_joint[j] = self.joints[self.child_body_to_joint[j]].getAnchorNumpy() /2 + self.joints[self.child_body_to_joint[j]].getAnchor2Numpy() /2

    ##---------------update anchor pos and muscle length -------------------##
    def update_anchor_pos(self):
        body0_rot = self.body_info.get_body_quat()[0]
        geom0_body0_offset = - R.from_quat(body0_rot).apply(self.geom0_body0_offset_tpose)
        geom0_gid = self.geoms0_gid[0]
        self.anchor_positions, muscle_length = self.world.cython_update_anchor_pos(self.anchor_local_postions[:, :, 0].reshape(-1), 
                                                                                    self.anchor_local_postions[:, :, 1].reshape(-1), 
                                                                                    self.anchor_body0_geom0_gid, 
                                                                                    self.anchor_body1_geom0_gid, 
                                                                                    self.anchor_act_force_weights, 
                                                                                    geom0_body0_offset, 
                                                                                    geom0_gid, 
                                                                                    self.muscle_hold_anchor_num)
        self.anchor_positions = self.anchor_positions.reshape([self.anchors_num, 3])
        return self.anchor_positions, muscle_length


    def update_anchor_pos_with_input_body_rot_pos(self, input_body_rot, input_body_pos):
        anchor_positions, muscle_length = self.world.cython_update_anchor_pos_with_input_body_rot_pos(input_body_rot.reshape(-1), 
                                                                                                      input_body_pos.reshape(-1), 
                                                                                                      self.geom_toward_body_rot_matrix.reshape(-1), 
                                                                                                      self.anchor_local_postions[:, :, 0].reshape(-1), 
                                                                                                      self.anchor_local_postions[:, :, 1].reshape(-1), 
                                                                                                      self.anchor_body0_idx, 
                                                                                                      self.anchor_body1_idx, 
                                                                                                      self.anchor_act_force_weights, 
                                                                                                      self.muscle_hold_anchor_num)
        return anchor_positions, muscle_length
    
    
    ##---------calculate muscle force and apply on bodies -------------------##
    def apply_pd_muscle_force(self, sim_muscle_len, ref_muscle_len):
        anchor_full, muscle_amp, muscle_passive_amp, unclip_muscle_amp =  self.world.cython_apply_pd_muscle_force(self.gamma, 
                                                                                                                    self.k_pe, 
                                                                                                                    self.e_mo, 
                                                                                                                    self.muscle_properties[:, 1].reshape(-1), 
                                                                                                                    self.muscle_properties[:, 2].reshape(-1), 
                                                                                                                    self.muscle_origin_length_calculated.reshape(-1), 
                                                                                                                    self.pose_aware_f0_max.reshape(-1), 
                                                                                                                    self.anchor_positions.reshape(-1), 
                                                                                                                    sim_muscle_len , 
                                                                                                                    ref_muscle_len, 
                                                                                                                    self.last_frame_sim_muscle_len.reshape(-1), 
                                                                                                                    self.kp_coff, 
                                                                                                                    self.kd_coff, 
                                                                                                                    self.fps, 
                                                                                                                    self.muscle_hold_anchor_num.reshape(-1),
                                                                                                                    self.muscle_fatigued_proportion.reshape(-1), 
                                                                                                                    self.muscle_activated_proportion.reshape(-1), 
                                                                                                                    self.muscle_resting_proportion.reshape(-1), 
                                                                                                                    self.LD_para, 
                                                                                                                    self.LR_para, 
                                                                                                                    self.F_para)
        self.world.cython_add_muscle_anchor_forces_to_body(self.map_anchor_to_act_force_body_cid, 
                                                            self.anchor_positions.reshape(-1), 
                                                            anchor_full.reshape(-1))
        self.last_frame_sim_muscle_len = sim_muscle_len
        return anchor_full, muscle_amp, muscle_passive_amp, unclip_muscle_amp 

    

    ##---------calculate hill type muscle activation -------------------##
    def calc_hill_type_activation(self, muscle_force, muscle_length):
        muscle_activation, un_clip_muscle_activation = self.world.cython_calc_hill_type_activation(self.gamma, 
                                                                                                    self.k_pe, 
                                                                                                    self.e_mo,  
                                                                                                    self.muscle_properties[:, 1].reshape(-1), 
                                                                                                    self.muscle_properties[:, 2].reshape(-1), 
                                                                                                    self.muscle_origin_length_calculated.reshape(-1), 
                                                                                                    self.pose_aware_f0_max.reshape(-1), 
                                                                                                    self.fps,muscle_force.reshape(-1), 
                                                                                                    muscle_length.reshape(-1),  
                                                                                                    self.last_frame_sim_muscle_len.reshape(-1), 
                                                                                                    self.muscle_fatigued_proportion.reshape(-1), 
                                                                                                    self.muscle_activated_proportion.reshape(-1), 
                                                                                                    self.muscle_resting_proportion.reshape(-1), 
                                                                                                    self.LD_para, 
                                                                                                    self.LR_para, 
                                                                                                    self.F_para)
        return muscle_activation, un_clip_muscle_activation


    ##---------muscle fatigue dynamics -------------------##
    def prepare_3cc_fatigue_model(self, max_force_txt_file):
        self.cython_calc_part_fatigue_weight(max_force_txt_file)
        self.calc_integral_3cc_model()

    def prepare_3cc_fatigue_model_with_input_mode(self, mode_idx, max_force_txt_file=None):
        if mode_idx == 0:
            # use the avg mode to calc part fatigue weight
            self.cython_calc_part_fatigue_weight_avg()
        elif mode_idx == 1:
            self.cython_calc_part_fatigue_weight_MASS_f0()
        elif mode_idx == 2:
            self.cython_calc_part_fatigue_weight(max_force_txt_file)
        else:
            raise NotImplementedError
        
        self.calc_integral_3cc_model()


    # 3CC-fatigue
    def cython_calc_part_fatigue_weight(self, max_force_txt_file):
        self.muscle_maximum_force_over_bvh = np.loadtxt(max_force_txt_file)
        hand_picked_muscle_maximum_force_over_bvh = np.ones(284) * 100.0
        self.cython_pick_max_muscle_force(hand_picked_muscle_maximum_force_over_bvh)
        self.part_fatigue_weight = self.world.cython_calc_part_fatigue_weight(self.muscle_maximum_force_over_bvh, self.muscle_part_idx)

    def cython_calc_part_fatigue_weight_MASS_f0(self):
        self.part_fatigue_weight = self.world.cython_calc_part_fatigue_weight(self.muscle_properties[:, 0].reshape(-1), self.muscle_part_idx)

    
    def cython_calc_part_fatigue_weight_avg(self):
        self.part_fatigue_weight = self.world.cython_calc_part_fatigue_weight_avg(self.muscle_part_idx)

    def calc_integral_3cc_model(self):
        time_horizon = 40.0
        force_load_horizon = 20.0
        force_load = 0.5
        sim_hz = self.sim_hz
        self.integral_3cc_model_activated, self.integral_3cc_model_resting, self.integral_3cc_model_fatigued = self.world.cython_calc_integral_3cc_model(force_load, 
                                                                                                                                                         time_horizon, 
                                                                                                                                                         force_load_horizon, 
                                                                                                                                                         sim_hz, 
                                                                                                                                                         self.r_para, 
                                                                                                                                                         self.F_para, 
                                                                                                                                                         self.R_para, 
                                                                                                                                                         self.LR_para, 
                                                                                                                                                         self.LD_para)

    


    def fatigue_step(self, calc_activation):
        """
        @parameters:
        M_A: instant muscle activated proportion
        M_R: instant muscle resting proportion
        M_F: instant muscle fatigued proportion
        target_load: need to calculate in this function, using M_A_next to calculate M_R_next, M_F_next
        fps: the env simulation Hz, using for integral in 3CC model
        r_para: 3CC model parameter
        F_para: 3CC model parameter
        R_para: 3CC model parameter
        LR_para: 3CC model parameter
        LD_para: 3CC model parameter
        -----------------------------
        muscle max force weight: use muscle max force as weight when calculating the part fatigue properties
        muscle part idx: muscle is in which part
        @returns:
        M_A: 3CC model next simulation step muscle activated proportion
        M_R: 3CC model next simulation step muscle resting proportion
        M_F: 3CC model next simulation step muscle fatigued proportion
        -----------------------------
        five body part together fatigue properties [0:Body, 1:Right_Leg, 2:Left_Leg, 3:Right_Arm, 4:Left_Arm]
        part_M_A: 3CC model next simulation step body-part muscle activated proportion
        part_M_R: 3CC model next simulation step body-part muscle resting proportion
        part_M_F: 3CC model next simulation step body-part muscle fatigued proportion
        """
        self.muscle_activated_proportion, self.muscle_resting_proportion, self.muscle_fatigued_proportion = self.world.cython_fatigue_step(self.muscle_activated_proportion, 
                                                                                                                                            self.muscle_resting_proportion, 
                                                                                                                                            self.muscle_fatigued_proportion, 
                                                                                                                                            calc_activation.reshape(-1), 
                                                                                                                                            self.sim_hz, self.r_para, 
                                                                                                                                            self.F_para, 
                                                                                                                                            self.R_para, 
                                                                                                                                            self.LR_para, 
                                                                                                                                            self.LD_para)
        self.part_activated_proportion, self.part_resting_proportion, self.part_fatigued_proportion = self.world.cython_part_fatigue_properties(self.muscle_activated_proportion, 
                                                                                                                                                self.muscle_resting_proportion, 
                                                                                                                                                self.muscle_fatigued_proportion, 
                                                                                                                                                self.part_fatigue_weight, 
                                                                                                                                                self.muscle_part_idx)

    def fatigue_state_random_start(self, mode_idx):
        target_load = 0.5 
        if mode_idx == 0:
            # using 5 body part as random granularity
            random_time = np.random.rand(5) * 10.0
            self.part_activated_proportion, self.part_resting_proportion, self.part_fatigued_proportion, self.muscle_activated_proportion, self.muscle_resting_proportion, self.muscle_fatigued_proportion = self.world.cython_3cc_fatigue_integral_part_fatigue_properties(self.muscle_part_idx, 
                                                                                                                                                                                                                                                                            target_load, 
                                                                                                                                                                                                                                                                            random_time, 
                                                                                                                                                                                                                                                                            self.sim_hz, 
                                                                                                                                                                                                                                                                            self.r_para, 
                                                                                                                                                                                                                                                                            self.F_para, 
                                                                                                                                                                                                                                                                            self.R_para, 
                                                                                                                                                                                                                                                                            self.LR_para, 
                                                                                                                                                                                                                                                                            self.LD_para, 
                                                                                                                                                                                                                                                                            self.integral_3cc_model_activated, 
                                                                                                                                                                                                                                                                            self.integral_3cc_model_resting, 
                                                                                                                                                                                                                                                                            self.integral_3cc_model_fatigued)
        elif mode_idx == 1:
            # using muscle group as random granularity
            random_time = np.random.rand(self.group_num) * 10.0
            self.part_activated_proportion, self.part_resting_proportion, self.part_fatigued_proportion, self.muscle_activated_proportion, self.muscle_resting_proportion, self.muscle_fatigued_proportion = self.world.cython_3cc_fatigue_integral_part_fatigue_properties_group_randomness(self.muscle_part_idx, 
                                                                                                                                                                                                                                                                                             target_load, 
                                                                                                                                                                                                                                                                                             random_time, 
                                                                                                                                                                                                                                                                                             self.sim_hz, 
                                                                                                                                                                                                                                                                                             self.r_para, 
                                                                                                                                                                                                                                                                                             self.F_para, 
                                                                                                                                                                                                                                                                                             self.R_para, 
                                                                                                                                                                                                                                                                                             self.LR_para, 
                                                                                                                                                                                                                                                                                             self.LD_para, 
                                                                                                                                                                                                                                                                                             self.integral_3cc_model_activated, 
                                                                                                                                                                                                                                                                                             self.integral_3cc_model_resting, 
                                                                                                                                                                                                                                                                                             self.integral_3cc_model_fatigued, 
                                                                                                                                                                                                                                                                                             self.muscle_in_which_group.reshape(-1), 
                                                                                                                                                                                                                                                                                             self.group_num, 
                                                                                                                                                                                                                                                                                             self.part_fatigue_weight.reshape(-1))
        else:
            raise NotImplementedError


    def cython_build_coact_matrix(self):
        self.group_avg_coact_matrix = self.world.cython_build_coact_matrix(self.muscle_groups_num, 
                                                                           self.muscle_in_which_group.reshape(-1), 
                                                                           self.pose_aware_f0_max.reshape(-1))
        return self.group_avg_coact_matrix

    def cython_build_coact_matrix_avg(self):
        self.group_avg_coact_matrix = self.world.cython_build_coact_matrix_avg(self.muscle_groups_num, 
                                                                               self.muscle_in_which_group.reshape(-1))
        return self.group_avg_coact_matrix

    ## ---------------auxiliary functions -------------------##
    def set_last_frame_muscle_len(self, input_muscle_len):
        self.last_frame_sim_muscle_len = input_muscle_len

    def cython_geom_pos(self):
        return self.world.geom_position(self.geoms0_gid)
    
    def scale_muscle_len(self, input_muscle_len):
        sclaed_muscle_len = self.world.cython_scale_calc_muscle_len(input_muscle_len.reshape(-1), 
                                                                    self.muscle_properties[:, 1].reshape(-1), 
                                                                    self.muscle_properties[:, 2].reshape(-1), 
                                                                    self.muscle_origin_length_calculated.reshape(-1))
        return sclaed_muscle_len
    
    def find_bodyId_by_name(self, body_name: str):
        idx = 0
        for body in self.bodies:
            if body.name == body_name:
                return idx
            idx += 1
        return None
    
    def cython_pick_max_muscle_force(self, instant_muscle_force):
        self.muscle_maximum_force_over_bvh = self.world.cython_pick_bigger_muscle_force(self.muscle_maximum_force_over_bvh, 
                                                                                        instant_muscle_force.reshape(-1))

    def reset_tpose(self):
        quat_action = np.zeros((self.bodies_num, 4), dtype=np.float64)
        new_body_pos, new_body_rot = self.forward_kinematics_with_local_action(quat_action)
        self.world.loadBodyPosODEFormat(self.body_info.body_c_id, new_body_pos.flatten())
        self.world.loadBodyQuatFromMatrix(self.body_info.body_c_id, new_body_rot.flatten())

    def set_body_pose_by_input(self, input_bd_rot, input_bd_pos):
        self.world.loadBodyPos(self.body_info.body_c_id, input_bd_pos.flatten())
        self.world.loadBodyQuat(self.body_info.body_c_id, input_bd_rot.flatten())

  
    def forward_kinematics_with_local_action(self, quat_action):
        """
        @output: body_pos [4*body_sz]
        @output: body_rot [12*body_sz]
        """
        new_body_pos, new_body_rot = self.world.cython_simple_forward_kinematics_with_local_action_target(self.bodies[0].PositionNumpy, 
                                                                                                          self.bodies[0].getRotationNumpy(), 
                                                                                                          quat_action.reshape(-1), 
                                                                                                          self.body_parent_index, 
                                                                                                          self.joint_tpose.reshape(-1), 
                                                                                                          self.body_com_tpose.reshape(-1), 
                                                                                                          self.child_body_to_joint_numpy.reshape(-1))
        return new_body_pos, new_body_rot