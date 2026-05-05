import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
import numpy as np
import networkx as nx

class RandomWalker:
    def __init__(self, L, num_envs, seed=0, n_x=10, n_a=5):
        self.L = L # grid size length
        self.num_envs = num_envs
        self.seed = seed
        self.n_x = n_x
        self.n_a = n_a
        self.G = nx.grid_2d_graph(L, L) # uses networkx to create a 2D grid graph
        self.A = nx.to_numpy_array(self.G) # converts the graph to a numpy adjacency matrix (row indicates a node, column indicates connection to other nodes)
        if self.n_a == 5: # give possible actions
            self.A = np.identity(self.L ** 2) + self.A # adding an identity matrix to the adjacency matrix which represents a self-loop on each node (to account for the "stay" action)
        D = np.diag(np.sum(self.A, axis=0)) # degree matrix (no. of connections for each node)

        # transition matrix (probability of moving from one node to another);
        # multiplying the inverse of the degree matrix with the adjacency matrix 
        # normalizes the rows of the adjacency matrix to sum to 1;
        # sum of each row in R is 1
        self.T = np.linalg.inv(D) @ self.A 
        self._init_world()

    def _init_world(self):
        np.random.seed(self.seed)
        
        valences = np.linspace(-1, 1, num=self.n_x) # get all valances in a specified range
        np.random.shuffle(valences) # randomize the order for observation assignment
        self.stimulus_to_valence = valences

        # the sensory stimuli are integers between 0 and n_x-1
        self.x = np.random.randint(0, self.n_x, size=(self.num_envs, self.L**2)) # set of sensory observations (for training)
        self.valid_x = np.random.randint(0, self.n_x, size=(self.num_envs, self.L**2)) # for validation

        self.x_vals = self.stimulus_to_valence[self.x]
        self.valid_x_vals = self.stimulus_to_valence[self.valid_x]

    def run(self, batch_size, seq_len, act_seed=1, validation=False, fix_start=False):
        """ Simulates the random walk and generates the data. 
        It returns 4 numpy arrays: pos (position), obs (observation), vals (valences), and act (action)"""
        np.random.seed(act_seed)
        obs = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
        vals = np.zeros((batch_size, seq_len + 1), dtype=np.float32)
        pos = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
        act = np.zeros((batch_size, seq_len), dtype=np.int32)
        env_ids = np.random.choice(np.arange(self.num_envs), batch_size, replace=True) # randomly selecting environments for each sequence in the batch
        cum_T = np.cumsum(self.T, axis=1)

        # get the first position and observation
        if fix_start:
            init_pos = np.zeros((batch_size, ), dtype=np.int32)
        else:
            init_pos = np.random.randint(0, self.L**2, size=(batch_size, 1))
        if validation:
            x = self.valid_x[env_ids]
            x_vals = self.valid_x_vals[env_ids]
        else:
            x = self.x[env_ids]
            x_vals = self.x_vals[env_ids]

        obs[:, 0] = np.take_along_axis(x, init_pos, axis=1)[:, 0] # efficient lookup for the observation at the initial position
        vals[:,0] = np.take_along_axis(x_vals, init_pos, axis=1)[:, 0] # same for valences
        pos[:, 0] = init_pos[:, 0]
        prev_pos = init_pos[:, 0]
        _rand = np.random.random((batch_size, seq_len, 1))

        for i in range(seq_len): # generating the walk (sequence of positions, observations and actions)
            _pos = (cum_T[prev_pos] < _rand[:, i]).sum(axis=1) # sampling the next position based on the transition probabilities from the current position
            pos[:, i + 1] = _pos
            obs[:, i + 1] = np.take_along_axis(x, _pos.reshape(batch_size, 1), axis=1)[:, 0] # same as above; efficient lookup of the observation at the new position
            vals[:, i + 1] = np.take_along_axis(x_vals, _pos.reshape(batch_size, 1), axis=1)[:, 0]
            tmp = _pos - prev_pos # action is the difference between the new position and the previous position
            # tmp=1 move east, tmp-1 move west, tmp+L move south, tmp-L move north
            if self.n_a == 5:
                # a=0 stay, a=1 east, a=2 south, a=3, north, a=4 west
                tmp = np.where(tmp == self.L, 2, tmp) # +L (south) -> actiom 2
                tmp = np.where(tmp == -self.L, 3, tmp) # -L (north) -> action 3
                act[:, i] = np.where(tmp == -1, 4, tmp) # -1 (west) -> action 4
            else:
                tmp = np.where(tmp == self.L, 2, tmp)
                tmp = np.where(tmp == - self.L, 3, tmp)
                act[:, i] = np.where(tmp == -1, 0, tmp) # -1 (west) -> action 0
            prev_pos = _pos # updating the previous position to the new position for next iteration
        return pos, obs, vals, act