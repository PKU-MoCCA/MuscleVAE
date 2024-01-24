from torch import nn
import torch
import MuscleVAECore.Utils.pytorch_utils as ptu
import MuscleVAECore.Utils.diff_quat as DiffRotation
from MuscleVAECore.Utils.motion_utils import *

def integrate_obs_vel(states, delta):
    shape_ori = states.shape

    if len(states.shape) == 2:
        states = states[None]
    assert len(states.shape) == 3, "state shape error"

    if len(delta.shape) == 2:
        delta = delta[None]
    assert len(delta.shape) == 3, "acc shape error"
    
    assert delta.shape[-1] == 6
    assert states.shape[-1] == 13

    batch_size, num_body, _ = states.shape
    vel = states[..., 7:10].view(-1, 3)
    avel = states[..., 10:13].view(-1, 3)
    delta = delta.view(-1, 6)
    local_delta = delta[..., 0:3]
    local_ang_delta = delta[..., 3:6]
    root_rot = torch.tile(states[:, 0, 3:7], [1, 1, num_body]).view(-1, 4)
    true_delta = quat_apply(root_rot,local_delta)
    true_ang_delta = quat_apply(root_rot, local_ang_delta)
    next_vel = vel + true_delta
    next_avel = avel + true_ang_delta
    return next_vel, next_avel


def integrate_state(states, delta, dt):
    pos = states[..., 0:3].view(-1, 3)
    rot = states[..., 3:7].view(-1, 4)
    
    next_vel, next_avel = integrate_obs_vel(states, delta)
    next_pos = pos + next_vel * dt
    next_rot = quat_integrate(rot, next_avel, dt)
    
    batch_size, num_body, _ = states.shape
    
    next_state = torch.cat([next_pos.view(batch_size, num_body, 3),
                            next_rot.view(batch_size, num_body, 4),
                            next_vel.view(batch_size, num_body, 3),
                            next_avel.view(batch_size, num_body, 3)
                            ], dim=-1)

    return next_state.view(states.shape)

class SimpleWorldModel(nn.Module):
    def __init__(self, without_muscle_ob_size, ac_size, delta_size, env, statistics,muscle_ob_size=0, **kargs) -> None:
        super(SimpleWorldModel, self).__init__()
        self.env_step_mode = kargs['env_step_mode']
        self.env = env
        self.use_muscle_state = self.env.use_muscle_state
        input_dim = without_muscle_ob_size + ac_size + muscle_ob_size
        output_dim = delta_size+ muscle_ob_size
        self.delta_size = delta_size

        self.mlp = ptu.build_mlp(
            input_dim= input_dim,
            output_dim= output_dim,
            hidden_layer_num=kargs['world_model_hidden_layer_num'],
            hidden_layer_size=kargs['world_model_hidden_layer_size'],
            activation=kargs['world_model_activation']
        )
        self.obs_mean = nn.Parameter(ptu.from_numpy(statistics['obs_mean']), requires_grad= False)
        self.obs_std = nn.Parameter(ptu.from_numpy(statistics['obs_std']), requires_grad= False)
        self.delta_mean = nn.Parameter(ptu.from_numpy(statistics['delta_mean']), requires_grad= False)
        self.delta_std = nn.Parameter(ptu.from_numpy(statistics['delta_std']), requires_grad= False)
        
        self.dt = self.env.dt
        
        self.weight = {}
        for key,value in kargs.items():
            if 'world_model_weight' in key:
                self.weight[key.replace('world_model_weight_','')] = value
        
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        observation = ptu.normalize(observation, self.obs_mean, self.obs_std)
        return observation
    
    @staticmethod
    def integrate_state(states, delta, dt):
        pos = states[..., 0:3].view(-1, 3)
        rot = states[..., 3:7].view(-1, 4)
        batch_size, num_body, _ = states.shape

        vel = states[..., 7:10].view(-1, 3)
        avel = states[..., 10:13].view(-1, 3)

        root_rot = states[:, 0, 3:7].view(-1,1,4)        
        delta = delta.view(batch_size, -1, 3)
        
        true_delta = broadcast_quat_apply(root_rot, delta).view(batch_size, -1, 6)
        true_delta, true_ang_delta = true_delta[...,:3].view(-1,3), true_delta[...,3:].view(-1,3)
        
        next_vel = vel + true_delta
        next_avel = avel + true_ang_delta
        
        next_pos = pos + next_vel * dt
        next_rot = quat_integrate(rot, next_avel, dt)
        next_state = torch.cat([next_pos.view(batch_size, num_body, 3),
                                next_rot.view(batch_size, num_body, 4),
                                next_vel.view(batch_size, num_body, 3),
                                next_avel.view(batch_size, num_body, 3)
                                ], dim=-1)
        return next_state.view(states.shape)
    
    
    def loss_with_muscle_fatigue_state(self, pred, tar, pred_muscle_state, tar_muscle_state):
        pred_pos, pred_rot, pred_vel, pred_avel = decompose_state(pred)
        tar_pos, tar_rot, tar_vel, tar_avel = decompose_state(tar)

        weight_pos, weight_vel, weight_rot, weight_avel, weight_ms = self.weight[
            "pos"], self.weight["vel"], self.weight["rot"], self.weight["avel"], self.weight["ms_fatigue"]

        batch_size = tar_pos.shape[0]

        pos_loss = weight_pos * \
            torch.mean(torch.norm(pred_pos - tar_pos, p=1, dim=-1))
        vel_loss = weight_vel * \
            torch.mean(torch.norm(pred_vel - tar_vel, p=1, dim=-1))
        avel_loss = weight_avel * \
            torch.mean(torch.norm(pred_avel - tar_avel, p=1, dim=-1))
        
        ms_fatigue_loss = weight_ms * \
            torch.mean(torch.norm(pred_muscle_state - tar_muscle_state, p=1, dim=-1))

        pred_rot_inv = quat_inv(pred_rot)
        tar_rot = DiffRotation.flip_quat_by_w(tar_rot.view(-1,4))
        pred_rot_inv = DiffRotation.flip_quat_by_w(pred_rot_inv.view(-1, 4))
        dot_pred_tar = torch.norm( DiffRotation.quat_to_rotvec( DiffRotation.quat_multiply(tar_rot,
                                              pred_rot_inv) ), p =2, dim=-1)
        rot_loss = weight_rot * \
            torch.mean(torch.abs(dot_pred_tar))
        return pos_loss, rot_loss, self.dt * vel_loss, self.dt * avel_loss, ms_fatigue_loss
    
    def forward(self, state, action, muscle_state=None,**obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation_rigid' in obs_info:
                observation = obs_info['observation_rigid']
            else:
                observation = state2ob(state)
            n_observation = self.normalize_obs(observation)

            if self.use_muscle_state:
                n_observation = torch.cat([n_observation, muscle_state], dim=-1)
        
        muscle_action = action
        n_delta = self.mlp( torch.cat([n_observation, muscle_action], dim = -1) )
        if self.use_muscle_state:
            muscle_state = n_delta[:, self.delta_size:]
            muscle_state = torch.sigmoid(muscle_state)
            n_delta = n_delta[:, :self.delta_size]

        delta = ptu.unnormalize(n_delta, self.delta_mean, self.delta_std)
        state = self.integrate_state(state, delta, self.dt)
        return state, muscle_state
        
