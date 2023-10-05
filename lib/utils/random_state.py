import random
import torch
import numpy as np
import pickle

def get_random_state():
    random_state = random.getstate()
    torch_state=torch.get_rng_state()
    np_state=np.random.get_state()
    states={'random':random_state,'torch':torch_state,'numpy':np_state}
    return states

def save_random_state(path):
    random_state = random.getstate()
    torch_state=torch.get_rng_state()
    np_state=np.random.get_state()
    states={'random':random_state,'torch':torch_state,'numpy':np_state}
    with open(path, 'wb') as f:
        pickle.dump(states, f)

def restore_random_state(state):
    if isinstance(state, str):
        with open(state,'rb') as f:
            state = pickle.load(f)
    random.setstate(state['random'])
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])

def seed_all(base_seed = None, worker_id=0):
    if base_seed is None:
        seed = random.getrandbits(64)
        base_seed = random.getrandbits(64)
    else:
        seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    # from pytorch code
    def _generate_state(base_seed, worker_id):
        INIT_A = 0x43b0d7e5
        MULT_A = 0x931e8875
        INIT_B = 0x8b51f9dd
        MULT_B = 0x58f38ded
        MIX_MULT_L = 0xca01f9dd
        MIX_MULT_R = 0x4973f715
        XSHIFT = 4 * 8 // 2
        MASK32 = 0xFFFFFFFF
        
        entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
        pool = [0] * 4
        hash_const_A = INIT_A
        def hash(value):
            nonlocal hash_const_A
            value = (value ^ hash_const_A) & MASK32
            hash_const_A = (hash_const_A * MULT_A) & MASK32
            value = (value * hash_const_A) & MASK32
            value = (value ^ (value >> XSHIFT)) & MASK32
            return value

        def mix(x, y):
            result_x = (MIX_MULT_L * x) & MASK32
            result_y = (MIX_MULT_R * y) & MASK32
            result = (result_x - result_y) & MASK32
            result = (result ^ (result >> XSHIFT)) & MASK32
            return result

        # Add in the entropy to the pool.
        for i in range(len(pool)):
            pool[i] = hash(entropy[i])

        # Mix all bits together so late bits can affect earlier bits.
        for i_src in range(len(pool)):
            for i_dst in range(len(pool)):
                if i_src != i_dst:
                    pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

        hash_const_B = INIT_B
        state = []
        for i_dst in range(4):
            data_val = pool[i_dst]
            data_val = (data_val ^ hash_const_B) & MASK32
            hash_const_B = (hash_const_B * MULT_B) & MASK32
            data_val = (data_val * hash_const_B) & MASK32
            data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
            state.append(data_val)
        return state
    np.random.seed(_generate_state(base_seed, worker_id))
    

