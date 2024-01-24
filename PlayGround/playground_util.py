import torch
import MuscleVAECore.Utils.diff_quat as DiffRotation
import numpy as np

def get_facing(state):
    assert state.shape[-1] == 13
    assert len(state.shape) == 2
    rot = state[:,3:7]
    direction = torch.zeros([rot.shape[0],3], dtype = rot.dtype, device = rot.device)
    direction[:,2] = 1
    
    facing_direction = DiffRotation.quat_apply(rot, direction)
    axis1 = facing_direction[:,0].view(-1,1)
    axis2 = facing_direction[:,2].view(-1,1)
    facing_direction = torch.cat([axis1, torch.zeros_like(axis1), axis2], dim = -1)
    facing_direction = facing_direction/ torch.linalg.norm(facing_direction, dim = -1, keepdim = True)
    return facing_direction

def get_root_facing(state):
    return get_facing(state[:,0])

def state2speed(state, mass):
    vel = state[...,:,7:10]
    com_vel = torch.einsum('bnk,n->bk', vel, mass)
    return com_vel

def state2speed_numpy(state, mass):
    vel = state[...,:,7:10]
    com_vel = np.einsum('nk,n->k', vel, mass)
    return com_vel