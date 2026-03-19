import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader   

from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import make_blobs
from tqdm import tqdm
from functions_and_utils import *
from configs import *

def create_datasets_and_loaders(config, pattern=None):
    ''' Create the dataset and data loaders based on the config
    # The dataloader will return a list of three elements. 
    # First element is a tensor of shape ( batch_size, seq_len, 30)
    # Second element is a tensor of shape ( batch_size, seq_len) with llc ids
    # Third element is a tensor of shape ( batch_size, seq_len) with hlc ids
    '''
    if config.dataset_name == 'contextual_switching_task':
        dataset = TaskDataset(config.no_of_blocks, config)
        if pattern == None:
            dataset_test = TaskDataset(config.no_of_blocks, config)
        else:
            dataset_test = TaskDataset_tests(config.no_of_blocks, config, pattern)
    elif config.dataset_name == 'contextual_switching_task_hierarchical':
        dataset = TaskDataset_hierarchical(config.no_of_blocks, config)
        dataset_test = TaskDataset_hierarchical(config.no_of_blocks, config)
    elif config.dataset_name == 'contextual_switching_task_2D':
        dataset = TaskDataset2D(config.no_of_blocks, config)
        dataset_test = TaskDataset2D(config.no_of_blocks, config)
    elif config.dataset_name == 'seq_learn':
        dataset = seq_learnDataset(config)
        dataset_test = seq_learnDataset(config, ) # not really implemented yet

    else:
        raise ValueError(f'Dataset {config.dataset_name} not implemented')

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    
    return dataset, dataset_test, train_loader, test_loader



class TaskDataset(Dataset):
    ''' Contextual switching task dataset '''
    def __init__(self, no_of_blocks, config):
        self.num_blocks = no_of_blocks
        self.block_size = config.block_size
        self.latent_change_interval = config.latent_change_interval
        self.default_std = config.default_std
        self.high_level_latent_change_interval_in_blocks = config.high_level_latent_change_interval_in_blocks
        self.config = config
        self.latent_values = config.training_data_means
        self.high_level_latent_values = [1, 2]
        self.seed = config.env_seed
        self.rng = np.random.default_rng(self.seed)
        self.data_rng = np.random.default_rng(self.seed)
        self.use_high_task_structure = config.use_high_task_structure

        # Generate block sizes once and use them consistently
        self.block_sizes = self.generate_block_sizes()

        # Generate sequences
        self.latent_sequence = self.generate_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        self.data_sequence = self.generate_data_sequence()

        # if len(self.data_sequence) < config.seq_len:
        #     raise ValueError(f'Seq_len: {config.seq_len} is longer than available data points: {len(self.data_sequence)}. Consider increasing the number of blocks.')

    def __len__(self):
        return (len(self.data_sequence) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len
        data = self.data_sequence[start:end]
        latent = self.latent_sequence[start:end]
        high_level_latent = self.high_level_latent_sequence[start:end]
        data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
        latent = torch.tensor(latent, dtype=torch.float32).reshape(-1, 1)
        high_level_latent = torch.tensor(high_level_latent, dtype=torch.float32).reshape(-1, 1)

        return data, latent, high_level_latent

    def generate_block_sizes(self):
        ''' Generate block sizes based on the configuration '''
        if self.config.block_duration_distribution == 'geometric':
            min_block_size = max(1, int(self.block_size / 1.5))
            max_block_size = int(self.block_size * 2)
            block_sizes = []
            for _ in range(self.num_blocks):
                raw_block_size = self.rng.geometric(2 / self.block_size)
                raw_block_size = min_block_size + raw_block_size
                block_size = min(raw_block_size, max_block_size)
                block_sizes.append(block_size)
        elif self.config.block_duration_distribution in ['fixed_block_size', 'fixed']:
            block_sizes = [self.block_size] * self.num_blocks
        else:
            raise ValueError(f"Invalid block_duration_distribution: {self.config.block_duration_distribution}")
        return block_sizes

    def generate_latent_sequence(self):
        if self.config.start_always_on_the_same_block:
            self.latent = np.min(self.latent_values)
        else:
            self.latent = self.rng.choice(self.latent_values)
        latent_sequence = []
        for i, block_size in enumerate(self.block_sizes):
            options = [lv for lv in self.latent_values if lv != self.latent]
            if len(options) > 0:
                self.latent = self.rng.choice(options)
            else:
                # self.latent = self.latent
                pass
            latent_sequence.extend([self.latent] * block_size)
        return latent_sequence

    def generate_high_level_latent_sequence(self):
        high_level_latent_sequence = []
        for i, block_size in enumerate(self.block_sizes):
            if i % self.high_level_latent_change_interval_in_blocks == 0:
                high_level_latent = self.rng.choice(self.high_level_latent_values)
            high_level_latent_sequence.extend([high_level_latent] * block_size)
        return high_level_latent_sequence

    def generate_data_sequence(self):
        data_sequence = []
        for i, block_size in enumerate(self.block_sizes):
            block_idx = sum(self.block_sizes[:i])  # Start index for current block
            mean = self.latent_sequence[block_idx]
            std = self.default_std
            _seed = self.high_level_latent_sequence[block_idx]
            if self.use_high_task_structure:
                self.data_rng = np.random.default_rng(_seed + self.seed)
            block_data = self.data_rng.normal(mean, std, block_size)
            data_sequence.extend(block_data)

        return data_sequence

    def truncate_data_sequence(self, end=None):
        ''' Truncate the data sequence to the length of the block size + self.config.length_of_opposite_block_sequence '''
        if end is None:
            end = self.block_size + self.config.length_of_opposite_block_sequence
        self.data_sequence = self.data_sequence[:end]
        self.latent_sequence = self.latent_sequence[:end]
        self.high_level_latent_sequence = self.high_level_latent_sequence[:end]

class TaskDataset2D(Dataset):
    ''' Contextual switching task dataset with 2D input '''
    def __init__(self, no_of_blocks, config):
        self.num_blocks = no_of_blocks
        self.block_size = config.block_size
        self.latent_change_interval = config.latent_change_interval
        self.default_std = config.default_std
        self.high_level_latent_change_interval_in_blocks = config.high_level_latent_change_interval_in_blocks
        self.config = config
        self.seed = config.env_seed
        self.rng = np.random.default_rng(self.seed)
        self.data_rng = np.random.default_rng(self.seed)
        self.use_high_task_structure = config.use_high_task_structure

        # Build 2D latent means from permutations
        base_means = config.training_data_means  # e.g., [0.2, 0.8]
        self.latent_values = [np.array(pair) for pair in [
            [base_means[0], base_means[1]],
            [base_means[1], base_means[0]],
            [base_means[1], base_means[1]],
        ]]
        self.high_level_latent_values = [1, 2]

        # If using EM demo data, pre-compute the synthetic dataset and assign blocks to clusters.
        if self.config.use_EM_demo_data:
            # Generate EM demo synthetic dataset exactly as in the provided code.
            random_state = 42
            n_samples = 1000
            # Specify centers and std deviations.
            x_means = [-0.4, 0.5, 0.0]
            y_means = [0.0, 0.5, 0.8]
            x_stds = [0.15, 0.2, 0.1]
            y_stds = [0.15, 0.2, 0.1]
            centers = np.column_stack((x_means, y_means))
            cluster_std = np.column_stack((x_stds, y_stds))
            X_em, y_em = make_blobs(n_samples=n_samples, centers=centers, 
                                    cluster_std=cluster_std, random_state=random_state)
            # Group the data by cluster.
            self.em_data_by_cluster = {}
            unique_clusters = np.unique(y_em)
            for cl in unique_clusters:
                self.em_data_by_cluster[cl] = X_em[y_em == cl]
            # For each block, assign a cluster from the available ones.
            self.block_cluster_assignments = self.rng.choice(list(self.em_data_by_cluster.keys()),
                                                              size=self.num_blocks)
        # Generate block sizes once and use them consistently
        self.block_sizes = self.generate_block_sizes()
        # Generate sequences (latent and high-level) as before.
        self.latent_sequence = self.generate_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        # Generate data sequence (will use EM demo data if enabled)
        self.data_sequence = self.generate_data_sequence()

    def __len__(self):
        return (len(self.data_sequence) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len

        data = self.data_sequence[start:end]
        latent = self.latent_sequence[start:end]
        high_level_latent = self.high_level_latent_sequence[start:end]

        data = torch.tensor(data, dtype=torch.float32)             # shape: (seq_len, 2)
        latent = torch.tensor(latent, dtype=torch.float32).reshape(-1, 1)
        high_level_latent = torch.tensor(high_level_latent, dtype=torch.float32).reshape(-1, 1)

        return data, latent, high_level_latent

    def generate_block_sizes(self):
        if self.config.block_duration_distribution == 'geometric':
            min_block_size = max(1, int(self.block_size / 1.5))
            max_block_size = int(self.block_size * 2)
            block_sizes = []
            for _ in range(self.num_blocks):
                raw_block_size = self.rng.geometric(2 / self.block_size)
                raw_block_size = min_block_size + raw_block_size
                block_size = min(raw_block_size, max_block_size)
                block_sizes.append(block_size)
        elif self.config.block_duration_distribution in ['fixed_block_size', 'fixed']:
            block_sizes = [self.block_size] * self.num_blocks
        else:
            raise ValueError(f"Invalid block_duration_distribution: {self.config.block_duration_distribution}")
        return block_sizes

    def generate_latent_sequence(self):
        '''Generate sequence of 2D latent vectors and 1D transformed values'''
        if self.config.use_EM_demo_data:
            # When using EM demo data, latent sequence can be arbitrary or derived from chosen cluster.
            # Here, we simply assign a latent value as the cluster id (cast to float) for each sample.
            latent_sequence = []
            for block_size in self.block_sizes:
                # Placeholder latent: use the cluster id for the block.
                latent_sequence.extend([0.0] * block_size)
            return latent_sequence
        else:
            if self.config.start_always_on_the_same_block:
                current_latent = self.latent_values[0]
            else:
                current_latent = self.rng.choice(self.latent_values)
            latent_sequence = []
            for block_size in self.block_sizes:
                options = [lv for lv in self.latent_values if not np.array_equal(lv, current_latent)]
                if len(options) > 0:
                    current_latent = self.rng.choice(options)
                latent_value = float(np.dot(current_latent, np.array([1.0, 1.5])))  # simple linear transform
                latent_sequence.extend([latent_value] * block_size)
            return latent_sequence

    def generate_high_level_latent_sequence(self):
        high_level_latent_sequence = []
        for i, block_size in enumerate(self.block_sizes):
            if self.config.use_EM_demo_data:
                # For EM demo data, we can simply pass the assigned cluster id as high-level latent.
                high_level_latent = float(self.block_cluster_assignments[i])
            else:
                if i % self.high_level_latent_change_interval_in_blocks == 0:
                    high_level_latent = self.rng.choice(self.high_level_latent_values)
            high_level_latent_sequence.extend([high_level_latent] * block_size)
        return high_level_latent_sequence

    def generate_data_sequence(self):
        data_sequence = []
        # If EM demo data mode is enabled, sample data from the pre-stored clusters per block.
        if self.config.use_EM_demo_data:
            for i, block_size in enumerate(self.block_sizes):
                cluster = self.block_cluster_assignments[i]
                cluster_data = self.em_data_by_cluster[cluster]
                # Sample block_size points with replacement from the chosen cluster.
                idx = self.rng.integers(0, len(cluster_data), size=block_size)
                block_data = cluster_data[idx]
                data_sequence.extend(block_data)
            return data_sequence
        else:
            for i, block_size in enumerate(self.block_sizes):
                block_start = sum(self.block_sizes[:i])
                mean_idx = block_start
                latent_sum = self.latent_sequence[mean_idx]
                # Find corresponding 2D mean vector
                matching_means = [lv for lv in self.latent_values if np.isclose(np.dot(lv, np.array([1.0, 1.5])), latent_sum)]
                if len(matching_means) == 0:
                    raise RuntimeError(f"No matching latent vector found for latent sum {latent_sum}")
                mean_vector = matching_means[0]

                if self.use_high_task_structure:
                    seed_val = self.high_level_latent_sequence[mean_idx]
                    self.data_rng = np.random.default_rng(seed_val + self.seed)

                block_data = self.data_rng.normal(mean_vector, self.default_std, (block_size, 2))
                data_sequence.extend(block_data)
            return data_sequence

    def truncate_data_sequence(self, end=None):
        if end is None:
            end = self.block_size + self.config.length_of_opposite_block_sequence
        self.data_sequence = self.data_sequence[:end]
        self.latent_sequence = self.latent_sequence[:end]
        self.high_level_latent_sequence = self.high_level_latent_sequence[:end]


class TaskOODDataset(Dataset):
    ''' Out-of-distribution task dataset for testing '''
    def __init__(self, config):
        self.block_size = config.block_size
        self.default_std = config.default_std
        self.config = config
        self.mean_values = np.arange(-0.2, 1.3, 0.1)  # Generate mean values from -0.2 to 1.2
        self.high_level_latent_value = 0  # Constant high-level latent value
        self.seed = config.env_seed
        self.rng = np.random.default_rng(self.seed)

        # Generate data sequences
        self.data_sequence = self.generate_data_sequence()
        self.latent_sequence = self.generate_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()

    def __len__(self):
        return (len(self.data_sequence) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len
        data = self.data_sequence[start:end]
        latent = self.latent_sequence[start:end]
        high_level_latent = self.high_level_latent_sequence[start:end]
        data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
        latent = torch.tensor(latent, dtype=torch.float32).reshape(-1, 1)
        high_level_latent = torch.tensor(high_level_latent, dtype=torch.float32).reshape(-1, 1)

        return data, latent, high_level_latent

    def generate_latent_sequence(self):
        ''' Generate latent sequence corresponding to mean values for each block '''
        latent_sequence = []
        for mean in self.mean_values:
            latent_sequence.extend([mean] * self.block_size)
        return latent_sequence

    def generate_high_level_latent_sequence(self):
        ''' Generate constant high-level latent sequence for all blocks '''
        high_level_latent_sequence = []
        for _ in self.mean_values:
            high_level_latent_sequence.extend([self.high_level_latent_value] * self.block_size)
        return high_level_latent_sequence

    def generate_data_sequence(self):
        ''' Generate Gaussian data sequence for each block '''
        data_sequence = []
        for mean in self.mean_values:
            block_data = self.rng.normal(mean, self.default_std, self.block_size)
            data_sequence.extend(block_data)
        return data_sequence

class TaskDataset_hierarchical(Dataset):
    ''' Contextual switching task dataset with sinusoidal and parabolic patterns '''
    def __init__(self, no_of_blocks, config):
        self.config = config
        self.num_blocks = no_of_blocks
        self.block_size = config.block_size
        self.default_std = config.default_std

        self.low_level_latent_values = self.config.training_data_means
        self.high_level_latent_values = self.config.high_level_latent_values  # 1 for sinusoid, 2 for parabola (or any other value for a fixed random sequence)
        self.seed = config.env_seed
        self.rng = np.random.default_rng(self.seed)
        self.data_rng = np.random.default_rng(self.seed)
        self.independent_noise_rng = np.random.default_rng(self.seed)
        self.low_level_latent_sequence = self.generate_low_level_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        self.data_sequence = self.generate_data_sequence()

    def __len__(self):
        return (len(self.data_sequence) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len
        data = self.data_sequence[start:end]
        low_latent = self.low_level_latent_sequence[start:end]
        high_latent = self.high_level_latent_sequence[start:end]

        data = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
        low_latent = torch.tensor(low_latent, dtype=torch.float32).reshape(-1, 1)
        high_latent = torch.tensor(high_latent, dtype=torch.float32).reshape(-1, 1)

        return data, low_latent, high_latent

    def generate_low_level_latent_sequence(self):
        ''' Low-level latent changes every block '''
        latent_sequence = []
        current_latent = self.rng.choice(self.low_level_latent_values)
        for _ in range(self.num_blocks):
            # Switch latent value for every block
            options = [lv for lv in self.low_level_latent_values if lv != current_latent]
            if len(options) > 0: # no other options available
                current_latent = self.rng.choice(options)
            latent_sequence.extend([current_latent] * self.block_size)
        return latent_sequence

    def generate_high_level_latent_sequence(self):
        ''' High-level latent changes every 4 blocks '''
        high_level_latent_sequence = []
        current_latent = self.rng.choice(self.high_level_latent_values)
        for i in range(self.num_blocks):
            # Change high-level latent every 4 blocks
            if i % 4 == 0:
                current_latent = self.rng.choice([lv for lv in self.high_level_latent_values if lv != current_latent])
            high_level_latent_sequence.extend([current_latent] * self.block_size)
        return high_level_latent_sequence

    def generate_data_sequence(self):
        ''' Generate the data sequence based on the latent sequences '''
        data_sequence = []
        for i in range(self.num_blocks):
            block_start = i * self.block_size
            mean = self.low_level_latent_sequence[block_start]
            high_latent = self.high_level_latent_sequence[block_start]
            std = self.default_std
            block_data = self.generate_block_data(mean, std, high_latent)
            data_sequence.extend(block_data)
        return data_sequence

    def generate_block_data(self, mean, std, high_latent):
        ''' Generate data for a block based on the high-level latent (sinusoid or parabola) '''
        x = np.linspace(0, 1, self.block_size)
        noise = self.data_rng.normal(0, std, self.block_size)
        if high_latent == 1:  # Sinusoid
            pattern = 0.1 * (np.sin(2 * np.pi * x))
        elif high_latent == 2:  # Parabola
            pattern = 1.2 * ((x - 0.5) ** 2)
        elif high_latent == 3 or high_latent==4: # random pattern 
            pattern = self.independent_noise_rng.normal(0, 0.3, self.block_size)
        else:    # every thing else is a FIXED random sequence 
            noise_rng = np.random.default_rng(high_latent + self.seed)
            pattern = noise_rng.normal(0, self.config.high_level_variance, self.block_size)
        return mean + pattern + noise



class TaskDataset_tests(TaskDataset):
    ''' A task dataset that returns repeating trials of config.latent_values based on an "exposure" string. '''
    
    def __init__(self, config, exposure_trial='c2p+', exposure_block_length=20):
        self.exposure_trial = exposure_trial
        self.exposure_block_length = exposure_block_length
        super().__init__(no_of_blocks=1, config=config)  # Assuming 1 block for simplicity
        self.num_blocks = 1
        self.latent_values = config.training_data_means
        self.block_size = exposure_block_length
        self.latent_sequence = self.generate_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        self.data_sequence = self.generate_data_sequence()


    def generate_latent_sequence(self, exposure_trial=None):
        ''' generate a latent sequence based on the exposure_trial string '''
        latent_sequence = []
        if exposure_trial is None:
            exposure_trial = self.exposure_trial
        if exposure_trial == 'c1p+':
            latent_sequence = [self.latent_values[1]] * self.block_size
        elif exposure_trial == 'c1p-': # alternate between two values
            latent_sequence = [self.latent_values[0]] * self.block_size
        elif exposure_trial == 'c2p-':
            latent_sequence = [self.latent_values[(1+i) % 2] for i in range(self.block_size)]
        elif exposure_trial == 'c2p+': # alternate between two values
            latent_sequence = [self.latent_values[(i % 2)] for i in range(self.block_size)]
            
        return latent_sequence
    
    def generate_data_sequence(self, sequence_patten = None): # override the parent method and genereate based on the latent_sequence trial by trial
        data_sequence = []
        if sequence_patten is None: 
            sequence_patten = self.sequence_patten
        if sequence_patten == 'easy_transition':
            for i in range(self.block_size):
                block_idx = i
                mean = self.latent_sequence[block_idx]
                std = self.default_std
                _seed = self.high_level_latent_sequence[block_idx]
                if self.use_high_task_structure:
                    self.data_rng = np.random.default_rng(_seed + self.seed)
                trial_data = self.data_rng.normal(mean, std, 1)
                data_sequence.extend(trial_data)

                latent_sequence = []
                high_level_latent_sequence = []
                for i in range(self.block_size):
                    if i % 2 == 0:
                        latent_sequence.append(self.latent_values[0])
                        high_level_latent_sequence.append(1)
                    else:
                        latent_sequence.append(self.latent_values[1])
                        high_level_latent_sequence.append(2)

        elif sequence_patten == 'hard_transition':
            pass
        elif sequence_patten == 'probe_low_high':
            pass
        else:
            raise ValueError(f'Invalid sequence pattern {sequence_patten}')
        return data_sequence


class seq_learnDataset(Dataset):
    ''' Task from Beukers, A. O., Collin, S. H., Kempner, R. P., Franklin, N. T., Gershman, S. J., & Norman, K. A. Blocked training facilitates learning of multiple schemas.
    (previously named seq_learnDataset)
    https://osf.io/preprints/psyarxiv/9bptj
    '''
    def __init__(self,  config):
        self.no_of_blocks = config.no_of_blocks
        self.config = config
        self.block_size = config.block_size
        self.task_length = 6
        self.no_of_tasks = 2
        self.space_size = 10
        self.rng = np.random.RandomState(config.env_seed)
        self.shuffle_or_interleave = config.shuffle_or_interleave
        self.random_transition_shuffle_or_interleave = config.random_transition_shuffle_or_interleave
        self.states,  self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def __len__(self):
        return (self.no_of_blocks * self.block_size - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, idx):
        start = idx * self.config.stride
        end = start + self.config.seq_len
        data = self.data[start:end]
        states, high_level_latents, low_level_latents = zip(*data)
        states             = torch.from_numpy(np.stack(states, dtype=np.float32)).reshape(-1, self.space_size)
        high_level_latents = torch.from_numpy(np.stack(high_level_latents, dtype=np.float32)).reshape(-1, 1)
        low_level_latents  = torch.from_numpy(np.stack(low_level_latents, dtype=np.float32)).reshape(-1, 1)
        return states, high_level_latents, low_level_latents

    def generate_data(self):
        # loop through blocks 
        # then loop through tasks
        states = []
        low_level_latent_list = []
        high_level_latent_list = []

        for block in range(self.no_of_blocks):
            if self.shuffle_or_interleave == 'shuffle':
                high_level_latent = self.rng.choice(self.no_of_tasks)
            else: # interleaved or blocked
                high_level_latent = int(((block) % 2)  )  #self.rng.choice(self.no_of_tasks)
            if self.random_transition_shuffle_or_interleave == 'shuffle': 
                low_level_latents = self.rng.choice([0, 1], size=(self.block_size//self.task_length)+1)
                # low level is generated for the entire block. Dvidied by task length. so a value for every story sequence in the block.
            else: # interleaved or blocked
                low_level_latents = int(((block+1) % 2)) # I only use this for interleaved, block sizs 1
                # low_level_latents = int(((block+1) % 4) > 1) # alternate every two stories
                low_level_latents = [low_level_latents] * (self.block_size//self.task_length)

            block_counter = 0
            for low_level_latent in low_level_latents:
                task_sequence = self.generate_states(high_level_latent, low_level_latent)
                block_counter += len(task_sequence)
                if block_counter > self.block_size: # if the task sequence is longer than the block size, then cut it off
                    no_of_states_needed_to_fill_block = self.block_size-(block_counter-self.task_length)
                    states.extend(task_sequence[:no_of_states_needed_to_fill_block])
                    low_level_latent_list.extend([low_level_latent]*(no_of_states_needed_to_fill_block))
                    high_level_latent_list.extend([high_level_latent]*(no_of_states_needed_to_fill_block))
                    break
                else:
                    states.extend(task_sequence)
                    low_level_latent_list.extend([low_level_latent]*self.task_length)
                    high_level_latent_list.extend([high_level_latent]*self.task_length)

        # states are a list of integers, convert to one hot encoding of size self.space_size
        states = np.eye(self.space_size)[states] * self.config.observation_scale

        return states, low_level_latent_list, high_level_latent_list

    def generate_states(self, high_level_latent, low_level_latent):
        ''' Given a high level latent and low level latent, generate the sequence of integer states for the task'''
        task_sequence = []
        if high_level_latent == 0:
            if low_level_latent == 0 or self.config.seq_learn_use_deterministic_transition_2:
                task_sequence.extend([0, 1, 3, 5, 7, 9])
            else:
                task_sequence.extend([0, 1, 4, 6, 8, 9])
        else:
            if low_level_latent == 0 or self.config.seq_learn_use_deterministic_transition_2:
                task_sequence.extend([0, 2, 3, 6, 7, 9])
            else:
                task_sequence.extend([0, 2, 4, 5, 8, 9])

        return task_sequence                
    
