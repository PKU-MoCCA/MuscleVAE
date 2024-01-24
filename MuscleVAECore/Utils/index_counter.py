import numpy as np
class index_counter():
    def __init__(self, done_flag) -> None:
        self.done_flag = done_flag
        self.cur_frame = 0
    
    @staticmethod
    def sample_rollout(feasible_index, batch_size, rollout_length):
        """generate index for rollout sampling

        Args:
            feasible_index (np.ndarray): please make sure [i,i+rollout_length) is useful
            batch_size (int): nop
            rollout_length (int): nop
        """
        begin_idx = np.random.choice(feasible_index.flatten(), [batch_size,1])
        bias = np.arange(rollout_length).reshape(1,-1)
        res_idx = begin_idx + bias
        return res_idx
    

    @staticmethod
    def calculate_feasible_index(done_flag, rollout_length):
        res_flag = np.ones(done_flag.shape[0]+1).astype(int)
        terminate_idx = np.where(done_flag!=0)[0].reshape(-1,1)
        terminate_idx_plus_1 = terminate_idx + 1
        bias_plus_1 = np.arange(rollout_length+1).reshape(1,-1)
        terminate_idx = (terminate_idx_plus_1 - bias_plus_1)
        res_flag[terminate_idx.flatten()] = 0
        return np.where(res_flag)[0][1:-1]

    
    @staticmethod
    def calculate_beign_feasible_index(done_flag):
        terminate_idx = np.where(done_flag!=0)[0]
        terminate_idx_plus_2 = terminate_idx + 2
        terminate_idx_plus_2 = terminate_idx_plus_2[:-1]
        beign_feasible_index = [1]
        [beign_feasible_index.append(i) for i in terminate_idx_plus_2]
        return np.array(beign_feasible_index)
    
    @staticmethod
    def random_select(feasible_index, p = None):
        return np.random.choice(feasible_index, p = p)
    
    @staticmethod
    def begin_frame_random_select(begin_feasible_index):
        return np.random.choice(begin_feasible_index)