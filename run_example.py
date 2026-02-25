# %% 
''' 
a scratch script to train models and experiment. Trains a model and plots the behavior, latent dynamics, and gradients.
'''

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    
# supress all warnings and messages from matplotlib
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import numpy as np
import copy

import plot_style
plot_style.set_plot_style()

from functions_and_utils import *
from configs import *

from train_and_infer_functions import *

# config = ContextualSwitchingTaskConfig(experiment_to_run='tweaking')
# config = ContextualSwitchingTaskConfig(experiment_to_run='weight_grads_comp')

# this produce the figures for the paper
config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
# config = seq_learnConfig(experiment_to_run='few_long_blocks')    
# config.block_size = 6
# config.shuffle_or_interleave = 'shuffle'
# config.add_interleaved_phase = True
# config.add_blocked_phase = False

# Run with more latents just to demo that. 
config.latent_dims = [10]
if config.use_add_gating:
    config.input_size += np.prod(config.latent_dims) # update the input size to include the latent
config.LU_lr = 0.01


# config = ContextualSwitchingTaskConfig(experiment_to_run='rnn')
# config = ContextualSwitchingTaskConfig(experiment_to_run='stride_2')

error_threshold=0.15
config.default_std = 0.1
config.log_weights = False

# config.LU_lr = 0.8
# config.l2_loss = 0.0
# config.LU_optimizer = 'adam'  # adamW or adam or sgd. AdamW did absolutely nothing different in a quick test.

baseline_rnn = False
if baseline_rnn: 
    config.LU_lr = 0.0
    # config.seq_len = 50 # for meta-learning in training mode
    # config.blocked_phase_length = 400 # same amount of data as in main fig.
    # config.env_seed = 1
else:   
    # config.env_seed = 0
    pass

print('Running the model seed: ', config.env_seed)
config.save_model = False
config.load_saved_model = False
logger, model, config, figs = train_model(config, seed=config.env_seed, 
                    save_models=False, load_models=False,)

if config.dataset_name == 'seq_learn':
    fig = plot_seq_learn_behavior_and_overall_corrects(logger_train, config, include_gradients=True)
else:
    panel_order = ['task_illustration_and_hierarchies', 'behavior',  'latent', 'latent_2d', 'gradients', 'weights_grad_norm',]# 'loss'] # 'task_illustration_and_hierarchies',
    fig = plot_logger_panels(logger, config, panel_order,  x2=None, annotate_phases='behavior')


from functions_and_utils_2 import rasterize_and_save
#%%
# final figure
compact_fig_cosyne = True
if compact_fig_cosyne: # Cosyne
    ''' 
    Here LU_LR 0.8 looks like too much ataxia but NeuraGEM converges fast,
      0.5 has more stable Z if there is room to show more blocks. '''
    panel_order = ['behavior',  'latent_2d', 'gradients', 'weights_grad_norm', 'latent_effective_lr']# 'loss'] # 'task_illustration_and_hierarchies',
    panel_order = ['behavior',  'latent_2d', 'gradients',]# 'loss'] # 'task_illustration_and_hierarchies',
    if baseline_rnn: panel_order = ['behavior']
    fig = plot_logger_panels(logger, config, panel_order,
                x2=260, dpi=300, subplot_height=.8, width = 2.5,
    rasterize=True)
    
    print(f'figure saved to: {config.export_path}{config.dataset_name}_behavior_rasterized.pdf')
    rasterize_and_save(fname= f'{config.export_path}{config.dataset_name}_behavior_rasterized.pdf',
                rasterize_list=None, fig=fig, dpi=300,
                savefig_kw={'bbox_inches': 'tight', 'transparent': True})
    
# slides
longer_fig_for_slides = False
if longer_fig_for_slides:
    ''' Here LU_LR 0.8 looks like too much ataxia '''
    panel_order = ['behavior',  'latent_2d', 'gradients', 'weights_grad_norm',]# 'loss'] # 'task_illustration_and_hierarchies',
    panel_order = ['gradients', ]# 'loss'] # 'task_illustration_and_hierarchies',
    fig = plot_logger_panels(logger, config, panel_order,  x2=400, dpi=300)
    # set figsize to [8, 6]
    fig.set_size_inches([18/2.53, 1])
    plt.savefig(f'{config.export_path}{config.dataset_name}_behavior.pdf', bbox_inches='tight')
    #print file path
    print(f'figure saved to: {config.export_path}{config.dataset_name}_behavior.pdf')

# save
logger_train = logger
save_path = f'{config.export_path}{config.dataset_name}_seed_{config.env_seed}'
if config.save_model:
    logger_train.config = config
    torch.save(model, (f'{save_path}_model.pth'))
    torch.save(logger_train, (f'{save_path}_logger.pth'))
    print(f'Model saved to {save_path}')
    