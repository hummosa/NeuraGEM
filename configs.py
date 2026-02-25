import os
import torch
import numpy as np

class Config:
    def __init__(self):
        self._run_name = 'default'
        self.no_of_steps_in_latent_space = 1 
        self.no_of_steps_in_weight_space = 1

        self.env_seed = 1
        self.save_model = False 
        self.load_saved_model = False
        self.eval_z_space_interval = 0 # use 0 to skip. Used in freezing the model at intervals and evaluating what representations appear in Z.
        self.add_noise_to_input = False
        self.noise_std = 0.0

        # Latent and optimization config
        '''
        Setup: latent dims control the number of latent variables.
        latent_chunks control the number of chunks in the latent space.
        Each chunk is a separate latent variable softmaxed separately.
        latent_aggregation_op controls what operation to apply to each chunk GRADs in the time dimension.
        exponential_increase_steepness controls the steepness of the exponential increase in the latent space
        It is expected to be a list with a value for each chunk.
        The latent_aggregation_op is not critical for the main results, but can be tuned to control the dynamics
        of the model to modulate attention to more recent time steps. 
        '''
        self.LU_lr = 0.1
        self.LU_Adam_betas =  (0.9, 0.999)
        self.latent_type = '1d_latent'
        self.latent_dims = [2] 
        self.latent_chunks = 1
        # self.latent_aggregation_op = 'none'
        # self.latent_aggregation_op = 'average'
        self.latent_aggregation_op = 'exponential_increase'
        self.exponential_increase_steepness = [0, 40][:self.latent_chunks]
        self.exponential_increase_multipliers = [1, 1][:self.latent_chunks]
        # self.latent_activation = 'softmax'
        self.softmax_temp = 1
        # self.latent_activation = 'sigmoid'
        self.latent_activation = 'none'
        
        self.LU_momentum = 0.0 # this is used for SGD
        self.WU_momentum = 0.0
        self.l2_loss = 0 # 0.0001
        self.pass_previous_latent = True # the last latent from previous timestep is passed as the initial latent for current timestep.
        self.LU_optimizer = 'Adam'
        self.WU_optimizer = 'Adam'
        self.loss_reduction_LU = 'sum'
        self.loss_reduction_WU = 'sum'

        # Training Curriculum params
        self.add_passive_learning_phase = False
        self.passive_phase_length = 200
        self.add_interleaved_phase = True
        self.latent_updates_during_shuffle = True
        self.interleaved_phase_length = 500
        # self.interleaved_phase_length = 2000
        self.add_blocked_phase = True
        self.blocked_phase_length = 1000
        self.shuffle_or_interleave = 'interleave'
        self.random_transition_shuffle_or_interleave = 'shuffle' # 'shuffled' # 'interleaved'
        self._allow_latent_updates = True # Do not use. Only for consturccting experiments. Can set no of LU to 0 if need to shut down LU from config.

        # Model architecture config
        self.rnn_type = 'lstm' #'lstm' #'rnn' # LSTM used to enable 'meta-rnn' training or training RNN with long horizon input. Otherwise gradient vanishing/exploding. Hochroeiter 1997.
        self.WU_lr = 0.001
        self.predict_first_frame = True # Model tries to predict 'explain' even first frame. There is information in the first frame that can be used to update latent. Even if prediction is not truely possible, feeback is, based on first frame. 
        self.use_add_gating = False # combine latent with input
        self.use_mul_gating = True
        self.use_input_attention = False
        self.P_gates_bernoulli_prob = 0.3
        self.pre_gating = True
        self.post_gating = not self.pre_gating
        self.what_latent_to_use = 'self' # 'taskID' # taskID gives the model a ground truth context ID
        # self.what_latent_to_use = 'taskID'
        self.run_test_phase = False # not used until the neural modules experiments.

        self.save_latent_updates = False

        self.length_of_opposite_block_sequence =4 # for COIN task
        self.observation_scale = 1. 
        
        self.log_weights = False
        self.log_hidden_states = False
        self.log_end_weights = False
        
        # RL config
        self.rl_task = False
        self.use_COIN_channel_experiment= False
        self.add_washout_phase = False # for COIN memory update experiment.
        self.start_always_on_the_same_block = False

        self.log_initial_burn_in_timesteps = False # logger typically records only the last timestep, then shifts forward. the initial timesteps at the begining of a simulation would be exlucded. This allows them logged. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.export_folder = './exports/'
        self.export_path = ''

        self.initialize_common_config()

    def initialize_common_config(self):
        self.input_size = None  # To be defined in subclasses
        self.hidden_size = None  # To be defined in subclasses
        self.output_size = None  # To be defined in subclasses
        self.seq_len = None  # To be defined in subclasses
        self.stride = None  # To be defined in subclasses
        self.dataset_name = ''  # To be defined in subclasses

        self.no_of_blocks = 4
        self.batch_size = 1
        self.epochs = 1
        self.block_size = 100

    @property
    def run_name(self):
        return self._run_name
    
    @run_name.setter
    def run_name(self, value):
        self._run_name = value
        self.update_export_path()

    def update_export_path(self):
        self.export_path = f'{self.export_folder}{self.dataset_name}/{self._run_name}/'
        os.makedirs(self.export_path, exist_ok=True)

    def reconfigure_for_prediction(self, experiment_to_run):
        # Adjust parameters for prediction mode based on the experiment condition
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = 0
        if hasattr(self, 'task_length'):
            self.block_size = 20 * getattr(self, 'task_length', 1)
        self.no_of_blocks = 4
        self.update_export_path()


class   ContextualSwitchingTaskConfig(Config):
    def __init__(self, experiment_to_run='figure'):
        super().__init__()
        self.dataset_name = 'contextual_switching_task'
        self.experiment_to_run = experiment_to_run
        self.input_size = 1
        if self.use_add_gating:
            self.input_size += np.prod(self.latent_dims) # update the input size to include the latent

        self.hidden_size = 32
        self.output_size = 1
        self.seq_len = 10
        self.stride = 1
        self.task_length = 1
        self.passive_epochs = 1
        self.epochs = 3
        self.batch_size = 1
        self.no_of_blocks = 200
        self.block_size = 50
        self.use_high_task_structure = False
        self.latent_change_interval = 1
        self.high_level_latent_change_interval_in_blocks = 3
        self.training_data_means = [0.2, 0.8]#[0.1, 0.3]
        self.default_std = 0.1 #0.01
        self.start_always_on_the_same_block = True
        self.block_duration_distribution = 'fixed_block_size' # 'geometric'
        # challenge block:
        self.out_of_distribution_challenge = {
            'use_challenge': True,
            'block_no': 15,
            'duration': 50,
            'mean': 0.5,
            'std': 0.2,
        }
        self.pre_window = 3 # for error strip analysis
        self.post_window = 20

        self.update_export_path()

        if experiment_to_run in ['tweaking', 'figure', 'weight_grads_comp']:
            self.save_model = False
            self.seq_len = 4
            self.stride = 1
            self.what_latent_to_use = 'self'
            self.pass_previous_latent = True
            self.no_of_steps_in_latent_space = 1 if self.pass_previous_latent else 10
            self.LU_Adam_betas = (0.6, 0.7) # (0.9, 0.999)
            self.latent_aggregation_op = 'exponential_increase' #'last' #'average'
            # self.latent_aggregation_op = 'average'
            self.l2_loss = 0.0001 if self.pass_previous_latent else 0
            self.latent_activation = 'softmax'
            # self.training_data_means = [0.1, 0.3]
            # self.default_std = 0.01
            self.passive_epochs = 1
            self.add_passive_learning_phase = False #if self.run_not_debug else False
            self.passive_phase_length = 500  #if self.run_not_debug else  0
            # self.softmax_temp = 0.3
            self.blocked_phase_length = 1200  #if self.run_not_debug else 550
            self.hidden_size = 64
            self.P_gates_bernoulli_prob = 0.3
            self.use_input_attention = False
            # self.latent_activation = 'none'
            # self.loss_reduction_LU = 'mean'
            self.loss_reduction_LU = 'mean'
            self.loss_reduction_WU = 'mean'

            self.LU_optimizer = 'Adam'
            self.LU_lr = 0.5 if self.pass_previous_latent else 0.8
            self.LU_momentum = 0.

            self.WU_optimizer = 'Adam'
            self.WU_lr = .005
            self.epochs = 1
            self.block_size = 25 * self.task_length
            self.block_duration_distribution = 'geometric'
        if experiment_to_run == 'figure': # this is for the fig 1 of the paper
            self.pass_previous_latent = True
            self.no_of_steps_in_latent_space = 1 if self.pass_previous_latent else 2
            self.LU_Adam_betas = (0.6, 0.7) # (0.9, 0.999)
            self.latent_aggregation_op = 'exponential_increase' #'last' #'average'
            # self.latent_aggregation_op = 'average'
            self.l2_loss = 0.0001 if self.pass_previous_latent else 0
            self.l2_loss *= 2000 if self.LU_optimizer == 'SGD' else 1
            
            self.WU_lr = 0.001 # SLOW VERSION 0.0001
            self.exponential_increase_steepness = [2]
            self.LU_lr = 0.8  # SLOW VERSION 0.2
            self.blocked_phase_length = 850  #if self.run_not_debug else 550
            

    def reconfigure_for_prediction(self, experiment_to_run):
        # Adjust parameters for prediction mode based on the experiment condition
        self.what_latent_to_use = 'self' # switching to optimizing latent. Could use taskID instead.
        self.batch_size = 1
        self.epochs = 1
        self.limited_testing_samples_no = int(2000/ self.batch_size) # to collect the same no of samples from predict
        self.no_of_blocks = 12
        self.no_of_steps_in_weight_space = 0
        self.add_noise_to_input = False 


class SeqLearnConfig(Config):
    def __init__(self, experiment_to_run='few_long_blocks'):
        super().__init__()
        self.dataset_name = 'seq_learn'
        self.experiment_to_run = experiment_to_run
        self.input_size = 10
        if self.use_add_gating:    self.input_size += np.prod(self.latent_dims) # update the input size to include the latent
        self.hidden_size = 32
        self.output_size = 10
        self.stride = 1
        self.task_length = 6
        self.observation_scale = 1
        self.seq_learn_use_deterministic_transition_2 = False
        self.plot_diagnostic_plots = False
        self.plot_dynamic_optim_gif = False
        self.grad_model_type = 'none'

        if experiment_to_run == 'few_long_blocks':
            self.seq_len = 18
            self.stride = 1
            self.what_latent_to_use = 'self'
            self.pass_previous_latent = False
            self.no_of_steps_in_latent_space = 1 if self.pass_previous_latent else 10
            self.l2_loss = 0.00001 if self.pass_previous_latent else 0
            self.latent_activation = 'softmax'
            self.latent_aggregation_op = 'average'
            self.passive_epochs = 1
            self.add_passive_learning_phase = False
            self.add_interleaved_phase = True
            self.latent_updates_during_shuffle = False
            self.use_input_attention = False
            self.passive_phase_length = 500 if not self.use_input_attention else 2000
            # self.softmax_temp = 0.3
            self.blocked_phase_length = 1200
            self.interleaved_phase_length = self.blocked_phase_length
            # self.use_input_attention = False
            self.hidden_size = 32
            self.P_gates_bernoulli_prob = 0.5
            # self.latent_activation = 'none'
            self.loss_reduction_LU = 'mean'
            self.loss_reduction_WU = 'mean'

            self.LU_optimizer = 'Adam'
            self.LU_lr = 0.1
            self.LU_momentum = 0.

            self.WU_optimizer = 'Adam'
            self.WU_lr = .001
            self.epochs = 1
            self.batch_size = 1
            self.block_size = 20 * self.task_length
            self.add_noise_to_input = False
            self.noise_std = 0.0
            self.no_of_blocks = self.blocked_phase_length // self.block_size

        self.limited_testing_samples_no = int(1000 / self.batch_size)
        
        self.update_export_path()

    def reconfigure_for_prediction(self, experiment_to_run):
            # Adjust parameters for prediction mode based on the experiment condition
            self.what_latent_to_use = 'self'
            if experiment_to_run == 'cere':
                self.no_of_steps_in_weight_space = 0
                self.block_size = 20 * self.task_length
                self.testing_phase_length = 3000 if (not self.grad_model_debug) else 440
                self.no_of_blocks = int(self.testing_phase_length / self.block_size)

                
            elif experiment_to_run == 'few_long_blocks' or experiment_to_run == 'interleaved':
                # self.seq_len = 6
                # self.no_of_steps_in_latent_space = 1
                self.no_of_steps_in_weight_space = 0
                self.block_size = 20 * self.task_length
                self.no_of_blocks = 4
                
            
            self.limited_testing_samples_no = int(500 / self.batch_size)


# Backwards compatibility alias (old name -> new)
seq_learnConfig = SeqLearnConfig
CSWConfig = SeqLearnConfig

