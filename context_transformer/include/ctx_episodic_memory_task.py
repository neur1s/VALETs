import numpy as np

class CtxtMemoryTask:
    def __init__(self, num_contexts, num_observations, obs_dim=256, 
                 context_dim=512, valence_no = 10, obs_per_valence=3,
                 feature_type = 'random', seed=42, pool_seed=None):

        self.num_contexts = num_contexts
        self.num_observations = num_observations
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.feature_type = feature_type

        # use dedicated RNGs so callers can control seeding for different parts
        rng = np.random.RandomState(seed)

        # creating observation representations (i.e., dictionary)
        if feature_type == 'random':
            self.obs_features = rng.standard_normal((num_observations, obs_dim))
        else:
            self.obs_features = self._load_word2vec(num_observations, obs_dim) # in case we decide to use word2vec embeddings

        # context vectors
        self.context_library = rng.standard_normal((num_contexts, context_dim))

        # valence assignment
        self.obs_valences = np.linspace(-1, 1, valence_no)
        all_valences = np.repeat(self.obs_valences, obs_per_valence)

        # create padding in case there's not enough valences
        if len(all_valences) < num_observations:
            pad_len = num_observations - len(all_valences)
            # sample padding values from the existing valence bins
            padding = rng.choice(self.obs_valences, pad_len)
            all_valences = np.concatenate([all_valences, padding])

        rng.shuffle(all_valences) # to prevent neighboring observations from having similar valences
        self.obs_to_valence = all_valences[:num_observations]

        # contextual constraints
        # each context only contains a random subset of 20% of all observations, to prevent global memorization and encourage context usage
        pool_rng = np.random.RandomState(pool_seed if pool_seed is not None else seed)
        self.context_pools = [
            pool_rng.choice(num_observations, num_observations // 5, replace=False)
            for _ in range(num_contexts)
        ]

    def get_episode(self, batch_size, seq_len):
        batch_ctx_ids = np.random.choice(self.num_contexts, batch_size)

        context_vecs = self.context_library[batch_ctx_ids]
        obs_vectors = np.zeros((batch_size, seq_len, self.obs_dim))
        valences = np.zeros((batch_size, seq_len))
        obs_ids = np.zeros((batch_size, seq_len), dtype=np.int32)

        for i, ctx_id in enumerate(batch_ctx_ids):
            pool = self.context_pools[ctx_id]

            chosen_obs = np.random.choice(pool, seq_len, replace=True) # sample a sequence of observations from the context's pool
            
            obs_ids[i] = chosen_obs
            obs_vectors[i] = self.obs_features[chosen_obs]
            valences[i] = self.obs_to_valence[chosen_obs]
            

        return {"context_vecs": context_vecs, 
                "obs_vectors": obs_vectors, 
                "valences": valences, 
                "obs_ids": obs_ids}


class CtxtMemoryDataset:
    def __init__(self, num_contexts, num_observations, num_samples, seq_len,
                 obs_dim=256, context_dim=512, feature_type='random', seed=42, pool_seed=None):
        self.core = CtxtMemoryTask(num_contexts=num_contexts,
                                   num_observations=num_observations,
                                   obs_dim=obs_dim,
                                   context_dim=context_dim,
                                   feature_type=feature_type,
                                   seed=seed,
                                   pool_seed=pool_seed)
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.seed = int(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + int(idx))

        batch_ctx_id = rng.choice(self.core.num_contexts)
        pool = self.core.context_pools[batch_ctx_id]
        chosen_obs = rng.choice(pool, self.seq_len, replace=True)

        context_vec = self.core.context_library[batch_ctx_id].astype(np.float32)
        obs_ids = chosen_obs.astype(np.int64)

        return batch_ctx_id, obs_ids