import torch
from torch.utils.data import Dataset, DataLoader   
import torch.nn as nn

import os
import numpy as np
import matplotlib as mpl
import copy

from tqdm import tqdm
from functions_and_utils import criterion_self_fullfilling_prophecy
from datasets import CoinDataset_probe, CoinDataset_channel, TaskDataset


##### for train_model #####
from models import RNN_with_latent
from train_and_infer_functions import *
from tqdm import tqdm
from functions_and_utils import *
from datasets import *


def _prepare_batch_inputs(model, config, raw_inputs, batch_llcids):
    """Prepare tensors needed for forward passes under different gating modes."""
    inputs = raw_inputs.to(config.device)
    llcids = batch_llcids.to(config.device)

    if config.add_noise_to_input:
        noise = torch.randn_like(inputs) * config.noise_std
        gated_inputs = inputs + noise
    else:
        gated_inputs = inputs

    combined_input = model.combine_input_with_latent(
        gated_inputs,
        what_latent=config.what_latent_to_use,
        taskID=llcids.round(),
    )

    if config.predict_first_frame:
        zero_frame = torch.zeros_like(inputs[:, :1, :])
        core_inputs = torch.cat((zero_frame, inputs[:, :-1, :]), dim=1)
    else:
        core_inputs = inputs

    return combined_input, core_inputs, inputs, llcids

def train_model(config, seed=0, save_models=True, load_models=False,
                run_test_phase=True):
    figs = {} # if any figs need produced during training, pass here
    config.env_seed = seed
    # set pytorch seed to config.seed
    torch.manual_seed(config.env_seed)
    np.random.seed(config.env_seed)

    # check if the path exists
    if not os.path.exists(config.export_path):
        os.makedirs(config.export_path)
    save_path = f'{config.export_path}{config.dataset_name}_exp_{config.experiment_to_run}'

    logger_train = Logger()
    logger_train.config = config

    model_folder = os.path.join(config.export_folder, 'models')
    model_name = f'model_seed_{seed}_blocked_phase_length_{config.blocked_phase_length}_exp_{config.experiment_to_run}.pt'
    model_path = os.path.join(model_folder, model_name)
    if load_models:
        if os.path.exists(model_path):
            # model.load_state_dict(torch.load(model_path))
            # this did not work. Load model directly from file without state dict
            model = torch.load(model_path)
            print('Model loaded successfully')
        else:
            print('Model not found')
    else:
        model = RNN_with_latent(config).to(config.device)
        # model.output_layer.weight.data = model.output_layer.weight.data+ .7 # change the init magnitude of the output layer to start preds around 0.5
        criterion = nn.MSELoss(reduction='none')
        ############################## RUN THE MODEL ##############################
        config._allow_latent_updates = False
        config.passive_epochs = 1
        if config.add_passive_learning_phase:
            logger_train.log_phase('no inference learning')
            config.no_of_blocks = int(config.passive_phase_length / config.block_size)
            dataset, dataset_test, dataloader, dataloader_test = create_datasets_and_loaders(config)

            print('Passive learning behavior')
            predictive_learning(logger_train, config, dataloader, model, criterion, epochs=config.passive_epochs)
            # fig = plot_behavior(explore_data_container, logger_train, config, print_shapes=False)#, x2=3000)
            logger_train.others['timestep_passive_learning_ended'] = len(logger_train.inputs)
        else:
            logger_train.others['timestep_passive_learning_ended'] = 0

        # logger_train = Logger_trainlogger_train() # reset the logger_train
        config._allow_latent_updates = True
        config.no_of_blocks = int(config.blocked_phase_length / config.block_size)

        dataset, dataset_test, dataloader, dataloader_test = create_datasets_and_loaders(config)
        # print('task length: ', config.task_length)
        # print('Blocked learning length: ', config.blocked_phase_length)
        # print('no of blocks: ', config.no_of_blocks)
        # print('block size: ', config.block_size)
        logger_train.log_phase('Learning and inference')
        predictive_learning(logger_train, config, dataloader, model, criterion)
        logger_train.others['timestep_learning_ended'] = len(logger_train.inputs)

        if run_test_phase:
            config.reconfigure_for_prediction(config.experiment_to_run)
            dataset, dataset_test, dataloader, dataloader_test = create_datasets_and_loaders(config)
            logger_train.log_phase('Inference only')
            predictive_learning(logger_train, config, dataloader_test, model, criterion)

            logger_train.log_phase('No inference nor learning')
            config.use_channel_experiment = False
            if config.use_channel_experiment:
                predictive_learning(logger_train, config, dataloader_test, model, criterion_self_fullfilling_prophecy)
            else:
                config.no_of_steps_in_latent_space = 0
                predictive_learning(logger_train, config, dataloader_test, model, criterion)


    if config.log_end_weights:
        # for mul_gating models, log P, the randomly chosen vectors
        if config.use_mul_gating:
            logger_train.others['P'] = model.P.detach().cpu().numpy()
        # if add_gating models log the input layer weights, Z is given as input, so Z to RNN hidden weights are part of the input layer weights
        logger_train.others['input_layer_weights'] = model.input_layer.weight.detach().cpu().numpy()

    if save_models:
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        # torch.save(model.state_dict(), model_path)
        torch.save(model, model_path)
    return  (logger_train, model, config, figs )


def run_generalized_tests(model, config, test_type='ood_means', weights_frozen=False):
    testing_loggers_dict = {}
    criterion = nn.MSELoss(reduction='none')

    if test_type == 'ood_means':
        values_to_test = np.round(np.arange(-0.2, 1.3, 0.1), 1)
    elif test_type == 'training_means':
        values_to_test = config.training_data_means
    elif test_type == 'ood_stds':
        values_to_test = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7,]
    elif test_type == 'block_size':
        values_to_test = [10, 12 ,13,  20, 30, 40, 50, 60,70,80]
    elif test_type == 'input_sweep':
        values_to_test = np.round(np.arange(-0.2, 1.3, 0.1), 1)
    else:
        raise ValueError('Invalid test_type specified')

    no_of_blocks = 2
    for value in values_to_test:
        logger = Logger()
        model_copy = copy.deepcopy(model)

        if test_type == 'ood_means' or test_type == 'training_means':
            training_data_means_saved = config.training_data_means
            config.training_data_means = [0.505, value] # 0.505 ends up being the pad_mean to give the models some data initially. Chosen to be a neutral middle value. the 0.005 is to differentiate it from the tested value 0.5
            if True: # not sure if these are bugs, I want to try this way
                config.training_data_means = [ value] 

        elif test_type == 'ood_stds':
            config.default_std = value
            no_of_blocks = 10
        elif test_type == 'block_size':
            config.block_size = value
            ### this ensures that really small blocks get more blocks. Small issue, hacky solution. Just to ensure enough data.
            baseline_block_size = 40
            default_no_of_blocks = 20
            if config.block_size < baseline_block_size:
                # Increase number of blocks by half the relative difference
                multiplier = 1 + 0.5 * ((baseline_block_size / config.block_size) - 1)
                no_of_blocks = int(default_no_of_blocks * multiplier)
            else:
                no_of_blocks = default_no_of_blocks
            config.block_duration_distribution = 'fixed_block_size'

        dataset = TaskDataset(no_of_blocks=no_of_blocks, config=config)

        if test_type in ['ood_means', 'block_size', 'ood_stds']: # this pads sequence just to get metarnns started with some data. The initial seq_len long data gets discarded otherwise and never logged.
            pad_length = config.seq_len + config.pre_window
            # pad_mean = 0.505
            first_mean = dataset.latent_sequence[0]
            other_means = [m for m in config.training_data_means if m != first_mean] 
            if len(other_means) == 0: # should never need this condition.
                other_means = [0.505] #config.training_data_means # 0.505 ends up being the pad_mean to give the models some data initially. Chosen to be a neutral middle value. the 0.005 is to differentiate it from the tested value 0.5
            pad_mean = np.random.choice(other_means)
            dataset.latent_sequence = [pad_mean] * pad_length + dataset.latent_sequence
            dataset.high_level_latent_sequence = dataset.high_level_latent_sequence[:pad_length] + dataset.high_level_latent_sequence
            while len(dataset.high_level_latent_sequence) < len(dataset.latent_sequence):
                dataset.high_level_latent_sequence.append(dataset.high_level_latent_sequence[-1])
            dataset.block_sizes.insert(0, pad_length)
            dataset.data_sequence = dataset.generate_data_sequence() # after updating the latents, now generate data from them.

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        model_copy.config.no_of_steps_in_weight_space = 0 if weights_frozen else 1
        config.no_of_steps_in_weight_space = 0 if weights_frozen else 1
        predictive_learning(logger, config, dataloader, model_copy, criterion)

        logger.config = config
        testing_loggers_dict[value] = logger

        if test_type == 'ood_means' or test_type == 'training_means':
            config.training_data_means = training_data_means_saved

    return testing_loggers_dict

def predictive_learning(logger, config, dataloader, model, criterion= nn.MSELoss(reduction='none'), epochs = None, grad_model=None):
    ''' Update after weights. Also detadch the latent before updating it. instead of reset '''
    training_losses_per_batch = []
    testing_loss = []
    training_losses_per_epoch = []
    
    epochs = config.epochs if epochs is None else epochs
    for epoch in range(epochs):
        # model.train()
        if config.rl_task:
            rewards, values, probs, ents = [], [], [], []
            accs = []
            env = dataloader.dataset
        running_loss = 0.0
        total_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=total_batches)
        # for bi, batch in pbar: # load just one batch for now
        #     inputs, batch_llcids, batch_hlcids = batch
        #     break
        for bi, batch in pbar:  
            inputs, batch_llcids, batch_hlcids = batch
            # if config.use_COIN_channel_experiment and batch_llcids[0, 0] == -2: # channel experiment, no feedback
            #     if config.stride == 1:
            #         criterion = criterion_self_fullfilling_prophecy
            #     else:
            #         config._original_latent_agg_op = config.latent_agg_op
            #         config.latent_agg_op = 'mask_last' # TODO address the concern that this is not returned back to its value.
            #         # this masks the second timestep which has the force field applied.

            combined_input, core_inputs, inputs, batch_llcids_device = _prepare_batch_inputs(
                model, config, inputs, batch_llcids
            )
            batch_llcids = batch_llcids_device
            batch_hlcids = batch_hlcids.to(config.device)
            if bi == 0 and config.log_initial_burn_in_timesteps: # this avoids missing the first seq_len timesteps when they are significant enough > 20
                for i in range(config.seq_len):
                    logger.log_input(np.expand_dims(inputs[:, i, :].cpu().detach().numpy(), axis =0))
            else:
                logger.log_input(inputs[:, -config.stride:, :].cpu().detach().numpy())
            model.WU_optimizer.zero_grad()
            model.LU_optimizer.zero_grad()

            if inputs.shape[1] == 1:
                pass
            # combined_input.detach_() # otherwise throws an error when backward pass is called. Update cannot have this. THen input which includes latent has no grad

            logger.log_training_batch(combined_input.cpu().detach().numpy()[:, -config.stride:, :])
            logger.llcids.append(batch_llcids[:, -config.stride:].cpu().detach().numpy())
            logger.hlcids.append(batch_hlcids[:, -config.stride:].cpu().detach().numpy())

             #### UPDATE the WEGIHTS ######################

            model.WU_optimizer.zero_grad()
            # Forward pass
            ###################################
            if config.use_add_gating:
                outputs, hidden_states = model(
                    combined_input,
                    taskID=batch_hlcids,
                    what_latent=config.what_latent_to_use,
                ) # taskID would be applied to latent if config.what_latent_to_use = 'taskID'
            elif config.use_mul_gating:
                outputs, hidden_states = model(
                    core_inputs,
                    taskID=batch_hlcids,
                    what_latent=config.what_latent_to_use,
                ) # apply_mul_gating will be called inside
            outputs = torch.stack(outputs, dim=1)
            # Log hidden states
            if config.log_hidden_states:
                logger.log_hidden_states(hidden_states) # detaching and cpuing moved to Logger obj
            ###################################
            # Compute the loss
            loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
            
            full_loss = loss.cpu().detach().numpy()
            logger.log_training_loss(full_loss[:, -config.stride:, :])

            # Backward pass and optimization
            loss = loss.sum() if config.loss_reduction_WU == 'sum' else loss.mean()
            loss.backward()
            weight_norms = nn = [torch.norm(param.grad).item() for name, param in model.named_parameters() if name in ['input_layer.weight', 'input_layer.bias', 'lstm_cell.weight_ih', 'lstm_cell.weight_hh', 'lstm_cell.bias_ih', 'lstm_cell.bias_hh', 'output_layer.weight', 'output_layer.bias']]
            logger.others['grad_norms'].append(np.mean(weight_norms))
            if config.no_of_steps_in_weight_space > 0:
                model.WU_optimizer.step()   
            elif config.no_of_steps_in_weight_space > 0:
                saved_grads = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        saved_grads.append(param.grad.clone())

            if bi == 0 and config.log_initial_burn_in_timesteps: # this avoids missing the first seq_len timesteps when they are significant enough > 20
                for i in range(config.seq_len):
                  logger.log_predicted_output(np.expand_dims(outputs.cpu().detach().numpy()[:, i, :], axis =0))
            else:
                logger.log_predicted_output(outputs.cpu().detach().numpy()[:, -config.stride:, :])
     
            if config.log_weights:
                logger.log_weights(model)
            logger.log_gradients_corrections(model.latent.grad.clone()[:, -config.stride:, :].cpu().detach().numpy())
            

            #### UPDATE the LATENT ######################
            if config.pass_previous_latent:
                model.detach_latent()
            else:
                model.reset_latent(batch_size = inputs.shape[0], seq_len = inputs.shape[1]) # pass input shape to re-init the latent in case dims have changed
            if config.no_of_steps_in_latent_space > 0 and config._allow_latent_updates:
                first_full_loss = model.update_latent(inputs,  criterion, logger, taskID = batch_hlcids,no_of_latent_steps = config.no_of_steps_in_latent_space, ) # log only the first batch to save memory

                logger.log_training_loss_before_latent_optimization(first_full_loss[:, -config.stride:, :])
            logger.log_latent_value(model.latent.clone()[:, -config.stride:, :].cpu().detach().numpy())
            # Log effective learning rate for the first latent dimension (Adam-specific)
            # Compute per-parameter effective LR: lr / (sqrt(v_hat) + eps), where v_hat is bias-corrected exp_avg_sq
            try:
                opt = getattr(model, 'LU_optimizer', None)
                if opt is not None:
                    # assume single param group / single parameter (Z)
                    group = opt.param_groups[0]
                    lr = float(group.get('lr', 0.0))
                    betas = group.get('betas', (0.9, 0.999))
                    eps = float(group.get('eps', 1e-8))
                    params = group.get('params', [])
                    if len(params) > 0:
                        p = params[0]
                        state = opt.state.get(p, None)
                        if state is not None and 'exp_avg_sq' in state and 'step' in state:
                            v = state['exp_avg_sq']
                            step = state.get('step', 0)
                            beta1, beta2 = betas
                            # avoid zero-division for step == 0
                            bias_correction2 = 1.0 - (beta2 ** step) if step > 0 else 1.0
                            # v_hat has same shape as Z (batch, seq_len, Z_dim)
                            v_hat = v / bias_correction2
                            effective_lr_tensor = lr / (v_hat.sqrt() + eps)
                            # take the last `config.stride` timesteps for the first latent dim
                            try:
                                if torch.is_tensor(effective_lr_tensor):
                                    eff_slice = effective_lr_tensor[:, -config.stride:, 0:1].cpu().detach().numpy()
                                else:
                                    eff_slice = np.array(effective_lr_tensor)[:, -config.stride:, 0:1]
                                # append an array of shape (batch, stride, 1) to follow other logging conventions
                                logger.others.setdefault('latent_effective_lr', []).append(eff_slice)
                            except Exception:
                                # fallback: skip logging if shapes/memory cause issues
                                pass
            except Exception:
                # non-fatal: if optimizer state isn't available or shape mismatches, skip logging
                pass
            if config.use_input_attention: logger.input_attention_weights.append(model.input_attention_weights.clone().cpu().detach().numpy())
            
        # Update the running loss
            running_loss += loss.item()
            training_losses_per_batch.append(loss.item())
            
            # this collects the z space evals during training for the figure comparing to EM dynamics
            if config.eval_z_space_interval and ( bi % config.eval_z_space_interval == 0):
                # pbar.set_description(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                ezsi = config.eval_z_space_interval
                config.eval_z_space_interval = 0
                zlogger, zdataset = eval_z_space(model, config, bi, )
                config.eval_z_space_interval = ezsi
                # save the logger
                folder = f'{config.export_path}{config.dataset_name}/z_space_evals/'
                os.makedirs(folder, exist_ok=True)
                np.save(folder + f'zlogger_{bi}.pkl', zlogger)

    # Print the average loss for the epoch
        average_loss = running_loss / len(dataloader)
        training_losses_per_epoch.append(average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')

def eval_z_space(model_org, config_org, bi):
    model = copy.deepcopy(model_org)
    config = copy.deepcopy(config_org)

    if config.dataset_name == 'contextual_switching_task':
        dataset = TaskDataset(no_of_blocks= 20, config=config)
    elif config.dataset_name == 'contextual_switching_task_2D':
        dataset = TaskDataset2D(no_of_blocks= 20, config=config)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    config.no_of_steps_in_weight_space = 0
    config.no_of_steps_in_latent_space = 5
    config.pass_previous_latent = False
    config.seq_len = 4

    model.config = config
    # config.stride = config.seq_len # so that every data point is evaluated only once.
    logger = Logger()
    predictive_learning(logger, config, dataloader, model, nn.MSELoss(reduction='none'), epochs = 1)

    return (logger, dataset)

