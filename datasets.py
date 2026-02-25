import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader   

from torch.utils.data import Dataset, DataLoader
import numpy as np
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
    elif config.dataset_name == 'tafazoli_task':
        dataset = TafazoliTaskDataset(config)
        dataset_test = TafazoliTaskDataset(config)
    elif config.dataset_name == 'coin':
        dataset = CoinDataset(config)
        dataset_test = CoinDataset(config)
    elif config.dataset_name == 'coin_spontaneous_recovery':
        dataset = CoinDataset_spontanuous_recovery(config)
        dataset_test = CoinDataset_spontanuous_recovery(config)
    elif config.dataset_name == 'hierarchical_reasoning':
        dataset = HierarchicalReasoningDataset(config)
        dataset_test = HierarchicalReasoningDataset(config, sensory_noise_type='none')

    else:
        raise ValueError(f'Dataset {config.dataset_name} not implemented')

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    
    return dataset, dataset_test, train_loader, test_loader

class CoinDataset(Dataset):
    ''' Adapted Dataset for a binary sequence prediction task with two timesteps per trial. '''
    def __init__(self, config):
        self.no_of_blocks = config.no_of_blocks
        self.config = config
        self.block_size = config.block_size
        self.task_length = 2  # Sequence length of 2 timesteps
        self.no_of_tasks = 2
        self.space_size = 2  # Binary states, so one-hot encoded vector of length 2
        self.rng = np.random.RandomState(config.env_seed)
        self.shuffle_or_interleave = config.shuffle_or_interleave
        self.states, self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def __len__(self):
        return (self.no_of_blocks * self.block_size - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, idx):
        start = idx * self.config.stride
        end = start + self.config.seq_len
        data = self.data[start:end]
        states, high_level_latents, low_level_latents = zip(*data)
        states = torch.from_numpy(np.stack(states, dtype=np.float32)).reshape(-1, self.space_size)
        high_level_latents = torch.from_numpy(np.stack(high_level_latents, dtype=np.float32)).reshape(-1, 1)
        low_level_latents = torch.from_numpy(np.stack(low_level_latents, dtype=np.float32)).reshape(-1, 1)
        return states, high_level_latents, low_level_latents

    def generate_data(self):
        states = []
        low_level_latent_list = []
        high_level_latent_list = []

        for block in range(self.no_of_blocks):
            if self.shuffle_or_interleave == 'shuffle':
                high_level_latent = self.rng.choice(self.no_of_tasks)
            else:
                high_level_latent = block % 2

            for _ in range(self.block_size // self.task_length):
                task_sequence = self.generate_states(high_level_latent)
                states.extend(task_sequence)
                high_level_latent_list.extend([high_level_latent] * self.task_length)
                low_level_latent_list.extend([0] * self.task_length)  # Unused in this version

        states = np.eye(self.space_size)[states] * self.config.observation_scale

        return states, high_level_latent_list, low_level_latent_list

    def generate_states(self, high_level_latent):
        ''' Generate a sequence of two states based on the task identifier '''
        if high_level_latent == 0:
            return [0, 0]
        else:
            return [1, 1]
class CoinDataset_spontanuous_recovery(Dataset):
    ''' Adapted Dataset for a binary sequence prediction task with two timesteps per trial. '''
    def __init__(self, config):
        self.no_of_blocks = 2
        self.config = config
        self.block_size = config.block_size
        self.task_length = 2  # Sequence length of 2 timesteps
        self.no_of_tasks = 2
        self.space_size = 2  # Binary states, so one-hot encoded vector of length 2
        self.length_of_opposite_block_sequence = config.length_of_opposite_block_sequence
        self.rng = np.random.RandomState(config.env_seed)
        self.shuffle_or_interleave = config.shuffle_or_interleave
        self.states, self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def __len__(self):
        return ((len(self.data)) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, idx):
        start = idx * self.config.stride
        end = start + self.config.seq_len
        data = self.data[start:end]
        states, high_level_latents, low_level_latents = zip(*data)
        states = torch.from_numpy(np.stack(states, dtype=np.float32)).reshape(-1, self.space_size)
        high_level_latents = torch.from_numpy(np.stack(high_level_latents, dtype=np.float32)).reshape(-1, 1)
        low_level_latents = torch.from_numpy(np.stack(low_level_latents, dtype=np.float32)).reshape(-1, 1)
        return states, high_level_latents, low_level_latents

    def generate_data(self):
        states = []
        low_level_latent_list = []
        high_level_latent_list = []

        high_level_latent = 0
        for _ in range(self.block_size // self.task_length):
            task_sequence = self.generate_states(high_level_latent)
            states.extend(task_sequence)
            high_level_latent_list.extend([high_level_latent] * self.task_length)
            low_level_latent_list.extend([0] * self.task_length)  # Unused in this version

        high_level_latent = 1
        for _ in range(self.length_of_opposite_block_sequence// self.task_length):
            task_sequence = self.generate_states(high_level_latent)
            states.extend(task_sequence)
            high_level_latent_list.extend([high_level_latent] * self.task_length)
            low_level_latent_list.extend([0] * self.task_length)

        # now the neutral null block, just need the input to be ambiguous at 0.5 0.5
        # high_level_latent = 0
        # for _ in range(self.block_size // self.task_length):
        #     task_sequence = self.generate_states(high_level_latent)
        #     states.extend(task_sequence)
        #     high_level_latent_list.extend([high_level_latent] * self.task_length)
        #     low_level_latent_list.extend([0] * self.task_length)

        states = np.eye(self.space_size)[states] * self.config.observation_scale
        # states[self.block_size+self.length_of_opposite_block_sequence:] = 0.5

        return states, high_level_latent_list, low_level_latent_list

    def generate_states(self, high_level_latent):
        ''' Generate a sequence of two states based on the task identifier '''
        if high_level_latent == 0:
            return [0, 0]
        else:
            return [1, 1]

class CoinDataset_wash_out(CoinDataset_spontanuous_recovery):
    ''' Adapted Dataset for a binary sequence prediction task with two timesteps per trial. '''
    def __init__(self, config, washout_block_length):
        self.washout_block_length = washout_block_length
        super().__init__(config)
        
        self.states, self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def generate_states(self, high_level_latent):
        ''' Generate a sequence of two states based on the task identifier '''
        if high_level_latent == 0:
            return [0, 0]
        else:
            return [1, 1]

    def generate_data(self):
        states = []
        low_level_latent_list = []
        high_level_latent_list = []

        for _ in range(self.washout_block_length):
            high_level_latent = np.random.choice([0, 1])
            task_sequence = self.generate_states(high_level_latent)
            states.extend(task_sequence)
            high_level_latent_list.extend([high_level_latent] * self.task_length)
            low_level_latent_list.extend([0] * self.task_length)

        states = np.eye(self.space_size)[states] * self.config.observation_scale

        states[1::2] = 0.5
      
        return states, high_level_latent_list, low_level_latent_list

class CoinDataset_gated_memory_update(CoinDataset_spontanuous_recovery):
    ''' Adapted Dataset for a binary sequence prediction task with two timesteps per trial. '''
    def __init__(self, config, exposure_trial= 'c2p+', test_block_length = 20):
        self.test_label = exposure_trial
        self.test_block_length = test_block_length
        super().__init__(config)
        
        self.states, self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def generate_states(self, high_level_latent, test_label):
        ''' Generate a sequence of two states based on the task identifier '''
        if test_label   == 'c1p+': return [0, 0]
        elif test_label == 'c1p-': return [0, 1]
        elif test_label == 'c2p+': return [1, 0]
        elif test_label == 'c2p-': return [1, 1]
        else: raise ValueError(f'Invalid test label {test_label}')

    def generate_data(self, ):
        states = []
        low_level_latent_list = []
        high_level_latent_list = []
        if self.test_label == 'c1p+' or self.test_label == 'c1p-':
            high_level_latent = 0
        elif self.test_label == 'c2p+' or self.test_label == 'c2p-':
            high_level_latent = 1

        for _ in range(self.test_block_length):
            task_sequence = self.generate_states(high_level_latent, test_label = self.test_label)
            states.extend(task_sequence)
            high_level_latent_list.extend([high_level_latent] * self.task_length)
            low_level_latent_list.extend([0] * self.task_length)
        states = np.eye(self.space_size)[states] * self.config.observation_scale                
        return states, high_level_latent_list, low_level_latent_list



class CoinDataset_probe(CoinDataset_spontanuous_recovery):
    ''' Adapted Dataset for a binary sequence prediction task with two timesteps per trial. '''
    def __init__(self, config, no_of_trials = 5):
        self.no_of_trials = max(no_of_trials, config.seq_len)
        super().__init__(config)
        self.states, self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def generate_data(self, ):
        ''' will use the low level latents to indicate probe trials where the loss from 
        the force p timestep should be masked as the model gets no feedback during those timesteps
        will use the llc value -2 to indicate probe trials
        '''
        states = []
        low_level_latent_list = []
        high_level_latent_list = []
        high_level_latent = 0
        probe_trials = self.no_of_trials
        for _ in range(probe_trials):
            task_sequence = self.generate_states(high_level_latent)
            states.extend(task_sequence)
            high_level_latent_list.extend([high_level_latent] * self.task_length)
            low_level_latent_list.extend([-2] * self.task_length)
        states = np.eye(self.space_size)[states] * self.config.observation_scale

        return states, high_level_latent_list, low_level_latent_list

class CoinDataset_channel(CoinDataset_spontanuous_recovery):
    ''' Channel trials offer no feedback. Will simply generate a sequence of 0.5s'''
    def __init__(self, config, channel_block_length):
        self.channel_block_length = channel_block_length
        super().__init__(config)
        self.states, self.high_level_latents, self.low_level_latents = self.generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def generate_data(self, ):
        ''' will use the low level latents to indicate probe trials where the loss from 
        the force p timestep should be masked as the model gets no feedback during those timesteps
        will use the llc value -2 to indicate probe trials
        '''
        states = []
        low_level_latent_list = []
        high_level_latent_list = []
        high_level_latent = 0
        probe_trials = max(self.channel_block_length, self.config.seq_len)
        for _ in range(probe_trials):
            task_sequence = self.generate_states(high_level_latent)
            states.extend(task_sequence)
            high_level_latent_list.extend([high_level_latent] * self.task_length)
            low_level_latent_list.extend([-1] * self.task_length)
        states = np.eye(self.space_size)[states] * self.config.observation_scale
        states = np.full((self.channel_block_length, 2), 0.5)

        return states, high_level_latent_list, low_level_latent_list
    


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


class TaskDataset_memory_update_exposure(TaskDataset):
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
        elif exposure_trial == 'c2p+': # alternate between two values
            latent_sequence = [self.latent_values[(i % 2)] for i in range(self.block_size)]
        elif exposure_trial == 'c1p-': # alternate between two values
            latent_sequence = [self.latent_values[(1+i) % 2] for i in range(self.block_size)]
        elif exposure_trial == 'c2p-':
            latent_sequence = [self.latent_values[0]] * self.block_size
            
        return latent_sequence
    
    def generate_data_sequence(self): # override the parent method and genereate based on the latent_sequence trial by trial
        data_sequence = []
        for i in range(self.block_size):
            block_idx = i
            mean = self.latent_sequence[block_idx]
            std = self.default_std
            _seed = self.high_level_latent_sequence[block_idx]
            if self.use_high_task_structure:
                self.data_rng = np.random.default_rng(_seed + self.seed)
            trial_data = self.data_rng.normal(mean, std, 1)
            data_sequence.extend(trial_data)
        return data_sequence

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
    

import numpy as np
import torch
from torch.utils.data import Dataset

class HierarchicalReasoningDataset(Dataset):
    ''' 
    Morteza Sarafyazd, Mehrdad Jazayeri, Hierarchical reasoning by neural circuits in the frontal cortex.
    Science 364, eaav8911 (2019). DOI: 10.1126/science.aav8911
    '''
    def __init__(self, config, sensory_noise_type=None):
        self.num_blocks = config.no_of_blocks
        self.block_size = config.block_size
        self.block_transition_function = config.block_transition_function
        self.config = config
        self.seed = config.env_seed
        self.rng = np.random.default_rng(self.seed)

        self.sensory_noise_type = config.sensory_noise_type if sensory_noise_type is None else sensory_noise_type
        self.sensory_noise_mag = config.sensory_noise_mag

        # Generate block sizes and sequences
        self.block_sizes = self.generate_block_sizes()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        self.states, self.latent_sequence = self.generate_latent_and_state_sequence()

    def __len__(self):
        return (len(self.states) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len
        states = torch.tensor(self.states[start:end], dtype=torch.float32).reshape(-1, 1)
        latent = torch.tensor(self.latent_sequence[start:end], dtype=torch.float32).reshape(-1, 1)
        high_level_latent = torch.tensor(self.high_level_latent_sequence[start:end], dtype=torch.float32).reshape(-1, 1)

        return states, latent, high_level_latent

    def generate_block_sizes(self):
        ''' Generate block sizes based on the specified transition function '''
        if self.block_transition_function == 'geometric':
            block_sizes = []
            remaining_trials = self.block_size * self.num_blocks
            while remaining_trials > 0:
                block_size = min(self.rng.geometric(0.5), remaining_trials)
                block_sizes.append(block_size)
                remaining_trials -= block_size
        elif self.block_transition_function == 'deterministic':
            block_sizes = [self.block_size] * self.num_blocks
        else:
            raise ValueError(f"Invalid block_transition_function: {self.block_transition_function}")
        return block_sizes

    def generate_high_level_latent_sequence(self):
        ''' Generate high-level latent sequence with alternating blocks '''
        high_level_latent_sequence = []
        current_latent = 0  # Alternates between 0 and 1

        for block_size in self.block_sizes:
            high_level_latent_sequence.extend([current_latent] * block_size)
            current_latent = 1 - current_latent  # Alternate between 0 and 1
        
        return high_level_latent_sequence

    def generate_latent_and_state_sequence(self):
        ''' Generate low-level latent sequence and stimulus-action pairs '''
        latent_sequence = []
        states = []

        '''
        In the study, the time interval \( t_s \) for trials during the monkeys' training was sampled from a uniform distribution
          between 530 and 1170 milliseconds. The sampled interval \( t_s \) was then used to determine whether it was shorter or longer than the median value of 850 milliseconds, which dictated the required response based on the current rule【6†source】. 
         '''

        # Generate stimulus times (ts) and threshold-based correct actions
        ts_values = (self.rng.uniform(530, 1170, size=sum(self.block_sizes)) - 530) / 1000
        ts_threshold = 0.85 - 0.53
        self.ts_threshold  = ts_threshold
        correct_actions = (ts_values < ts_threshold).astype(int)

        # Apply block-wise transformation based on high-level latent
        for i, block_size in enumerate(self.block_sizes):
            block_start = sum(self.block_sizes[:i])
            block_end = block_start + block_size
            if self.high_level_latent_sequence[block_start] == 1:
                correct_actions[block_start:block_end] = 1 - correct_actions[block_start:block_end]

        # Add sensory noise if needed
        if self.sensory_noise_type == 'gaussian':
            sensory_noise = self.rng.normal(0, self.sensory_noise_mag, size=sum(self.block_sizes))
        else:
            sensory_noise = np.zeros(sum(self.block_sizes))
        ts_values += sensory_noise

        # Construct stimulus-action sequence
        for i in range(0, len(ts_values), 2):  # Take every second trial
            if i + 1 < len(ts_values):
                states.append(ts_values[i])  # Stimulus
                states.append(correct_actions[i])  # Correct action
                latent_sequence.append(self.high_level_latent_sequence[i])  # Store block type
                latent_sequence.append(self.high_level_latent_sequence[i])  # Maintain sequence alignment

        return states, latent_sequence

    def generate_difficult_ts_blocks(self, latent_value, distance=0.1, block_size=100 ):
        """
        Generate stimulus times (ts) for a block with a given difficulty level, where 
        the ts values are close to the decision threshold to create ambiguity.
        
        Parameters:
        -----------
        distance: float
            Distance from the decision threshold.
            hardest is same as noise magnitude during training. 
        
        Returns:
        --------
        states : np.ndarray
            Sequence of stimulus times and correct actions.
        high_level_latent_sequence : np.ndarray
            The high-level latent sequence.
        latent_sequence : np.ndarray
            The latent sequence (same as high-level latent sequence).
        """
        threshold = 0.85 - 0.53  # Decision threshold (0.32)
        self.ts_threshold  = threshold
        half_block = block_size // 2  # Ensure even distribution above and below threshold

        # Generate ts values near the threshold
        ts_below = np.full(half_block, threshold - distance)  # Below threshold
        ts_above = np.full(half_block, threshold + distance)  # Above threshold
        
        # Combine and shuffle within block
        ts_block = np.concatenate([ts_below, ts_above])
        self.rng.shuffle(ts_block)  # Shuffle to avoid patterns

        # add some random wiggles, not noise, but wiggles.
        # ts_block+= self.rng.normal(0, self.sensory_noise_mag, size=len(ts_block))
        ts_block+= self.rng.normal(0, 0.0, size=len(ts_block))

        # Compute correct actions before flipping based on block latent
        correct_actions = (ts_block < threshold).astype(int)
        if latent_value ==1:
            correct_actions = 1 - correct_actions
        
        # Construct stimulus-action sequence
        states = []
        for i in range(0, len(ts_block), 2):  # Ensure proper stimulus-action pairing
            if i + 1 < len(ts_block):
                states.append(ts_block[i])  # Stimulus
                states.append(correct_actions[i])  # Correct action
        
        states = np.array(states)
        high_level_latent_sequence = np.full(len(states), latent_value)  # Single latent task, default to 0
        latent_sequence = high_level_latent_sequence.copy()
        
        return states, high_level_latent_sequence, latent_sequence


import gymnasium as gym
from gymnasium import spaces
from sklearn.datasets import make_blobs

class TafazoliTaskDataset(gym.Env):
    """
    Environment for classifying based on color or shape dimensions according to a rule.
    - Input: 2D vector, first element color, second element shape.
    - Output: 2D action vector. The action depends on the task rule.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = 2
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.input_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_trials =  config.no_of_blocks * config.block_size
        self.rule = 2
        self.block_size = config.block_size
        self.trial = 0
        self.state = None
        self.latent_values = config.latent_values
        self.high_level_latent_values = config.high_level_latent_values
        self.latent_change_interval = config.block_size
        self.high_level_latent_change_interval_in_blocks = config.high_level_latent_change_interval_in_blocks
        self.default_std = config.default_std
        self.seed = config.env_seed
        self.rng = np.random.default_rng(self.seed)
        self.data_rng = np.random.default_rng(self.seed)
        
        # Generate sequences
        self.latent_sequence = self.generate_latent_sequence()
        self.high_level_latent_sequence = self.generate_high_level_latent_sequence()
        self.state_sequence = self.generate_all_states(self.max_trials)
        self.state = self.state_sequence[0]

    def __len__(self):
        return (self.max_trials - self.config.seq_len) // self.config.stride + 1
    
    def __getitem__(self, idx):
        start = idx * self.config.stride
        end = start + self.config.seq_len
        states = self.state_sequence[start:end]
        latents = self.latent_sequence[start:end]
        high_level_latents = self.high_level_latent_sequence[start:end]
        states = torch.tensor(states, dtype=torch.float32).reshape(-1, 2)
        latents = torch.tensor(latents, dtype=torch.float32).reshape(-1, 1)
        high_level_latents = torch.tensor(high_level_latents, dtype=torch.float32).reshape(-1, 1)
        return states, latents, high_level_latents

    def step(self, action):
        self.trial += 1
        done = self.trial >= self.max_trials
        reward = 0
        info = {}

        # Switch rule every block_size trials
        if (self.trial % self.block_size) == 0:
            self.rule = (self.rule % 3) + 1  # Cycle through rules 1, 2, 3

        # Evaluate action based on the current rule
        self.rule = self.latent_sequence[self.trial]
        self.axis = self.high_level_latent_sequence[self.trial]
        reward = self.reward_function([action])
        correct_action = self.get_correct_action(state=self.state, rule=self.rule, axis=self.axis)

        # Generate new state
        self.state = np.random.choice(np.arange(-1, 1.2, 0.2), size=2)
        # add perceptual noise:
        perceptual_noise= self.data_rng.normal(0, self.default_std, size=2)
        self.state += perceptual_noise
        info.update({'rule': self.rule, 'axis': self.axis,
                    'correct_action': correct_action, 'perceptual_noise': perceptual_noise})
        return self.state, reward, done, info

    def reward_function(self, actions, states, rules, axes):
        rewards = []
        for action, state, rule, axis in zip(actions, states, rules, axes):
            correct_action = self.get_correct_action(state=state, rule=rule, axis=axis)
            # reward = 1 if action == correct_action else 0
            reward = np.array(1 if action == correct_action else 0)
            rewards.append(reward- 0.5)
        
        return rewards 

    def reset(self):
        self.trial = 0
        self.state = np.random.choice(np.arange(-1, 1.2, 0.2), size=2)
        self.rule = self.latent_sequence[self.trial]
        self.axis = self.high_level_latent_sequence[self.trial]
        reward, done, info = 0, False, {}
        return self.state, reward, done, info

    def generate_latent_sequence(self):
        self.latent = self.rng.choice(self.latent_values)
        latent_sequence = []
        for i in range((self.max_trials // self.block_size)+1): # +1 to include the last incomplete block
            # self.latent = self.rng.choice([lv for lv in self.latent_values if lv != self.latent])
            # go through 1, 2, 3 sequentially and repeat
            self.latent = self.latent_values[i % len(self.latent_values)]
            latent_sequence.extend([self.latent] * self.block_size)
            if len(latent_sequence) >= self.max_trials:
                latent_sequence = latent_sequence[:self.max_trials]
        return latent_sequence

    def generate_high_level_latent_sequence(self):
        high_level_latent_sequence = []
        for i in range((self.max_trials // self.block_size)+1): # +1 to include the last incomplete block
            if i % self.high_level_latent_change_interval_in_blocks == 0:
                # high_level_latent = self.rng.choice(self.high_level_latent_values)
                # go through 1, 2 sequentially and repeat
                high_level_latent = self.high_level_latent_values[i % len(self.high_level_latent_values)]
            high_level_latent_sequence.extend([high_level_latent] * self.block_size)
            if len(high_level_latent_sequence) >= self.max_trials:
                high_level_latent_sequence = high_level_latent_sequence[:self.max_trials]
        return high_level_latent_sequence

    def get_correct_action(self, state=None, rule=None, axis=None):
        if rule is None:
            rule = self.rule
        # if axis is None: # not really using it at the moment.
            # axis = self.axis
        if state is None:
            state = self.state
        rule = rule.squeeze()
        state = state.squeeze()
        correct_action = 0
        if rule == 1:
            correct_action = 1 if state[1] > 0 else 0
        elif rule == 2:
            correct_action = 3 if state[0] > 0 else 2
        elif rule == 3:
            correct_action = 1 if state[0] > 0 else 0
        return correct_action
    
    def generate_correct_actions(self):
        correct_actions = []
        for state, rule, axis in zip(self.state_sequence, self.latent_sequence, self.high_level_latent_sequence):
            correct_action = self.get_correct_action(state=state, rule=rule, axis=axis)
            correct_actions.append(correct_action)
        return correct_actions
    
    def generate_all_states(self, max_trials):
        states = []
        for _ in range(max_trials):
            state = np.random.choice(np.arange(-1, 1.2, 0.2), size=2)
            states.append(state)
        return states