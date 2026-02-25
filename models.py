import numpy as np
import torch
import torch.nn as nn
import copy
from typing import List, Optional, Tuple, Union

import math


class RNN_with_latent(nn.Module):
    """Recurrent model with an inference-time latent variable ``Z`` that gates the dynamics."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = getattr(config, "device", torch.device("cpu"))

        self.hidden_size = int(config.hidden_size)
        self.Z_dim = int(np.prod(config.latent_dims))
        self.latent_dim = self.Z_dim  # legacy name used elsewhere
        self.Z_chunks = max(int(getattr(config, "latent_chunks", 1)), 1)

        if self.Z_dim % self.Z_chunks != 0:
            raise ValueError(
                "latent_dims must be divisible by latent_chunks. "
                f"Got latent_dims={config.latent_dims} and latent_chunks={self.Z_chunks}."
            )

        self.use_add_gating = bool(getattr(config, "use_add_gating", False))
        self.use_mul_gating = bool(getattr(config, "use_mul_gating", False))

        self.input_layer = nn.Linear(config.input_size, self.hidden_size)
        self.rnn_cell = self._build_recurrent_cell(config.rnn_type)
        self.output_layer = nn.Linear(self.hidden_size, config.output_size)

        if self.use_mul_gating:
            self._build_static_gates()
        else:
            self.gating_mask1 = None
            self.gating_mask2 = None

        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None

        self.init_hidden()
        self.init_Z()

        self.loss_function = nn.MSELoss(reduction="none")

        self.exponential_increase_filter: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self.init_exponential_increase_filter(config)

        self.W_optimizer = self._build_W_optimizer()
        self._rebuild_Z_optimizer()
        # legacy aliases
        self.WU_optimizer = self.W_optimizer
        self.LU_optimizer = self.Z_optimizer

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ # 

    def _build_recurrent_cell(self, rnn_type: str) -> nn.Module:
        rnn_type = rnn_type.lower()
        if rnn_type == "lstm":
            return nn.LSTMCell(self.hidden_size, self.hidden_size)
        if rnn_type == "gru":
            return nn.GRUCell(self.hidden_size, self.hidden_size)
        if rnn_type == "rnn":
            return nn.RNNCell(self.hidden_size, self.hidden_size)
        raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Choose from lstm, gru, rnn.")

    def _build_static_gates(self) -> None:
        prob = float(getattr(self.config, "P_gates_bernoulli_prob", 0.5))
        prob = min(max(prob, 0.0), 1.0)
        shape = (self.Z_dim, self.hidden_size)

        gate1 = torch.bernoulli(torch.full(shape, prob, device=self.device))
        gate2 = torch.bernoulli(torch.full(shape, prob, device=self.device))

        self.register_buffer("gating_mask1", gate1)
        self.register_buffer("gating_mask2", gate2)
        # legacy alias used in training scripts
        self.P = self.gating_mask1

    # ------------------------------------------------------------------ #
    # Hidden state / Z state initialisation
    # ------------------------------------------------------------------ #

    def init_hidden(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = batch_size or int(getattr(self.config, "batch_size", 1))
        base_value = 10.0 / self.hidden_size
        self.hidden_state = torch.full((batch_size, self.hidden_size), base_value, device=self.device)

        if self.config.rnn_type.lower() == "lstm":
            self.cell_state = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            self.cell_state = None
        return self.hidden_state, self.cell_state

    def init_Z(
        self,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = batch_size or int(getattr(self.config, "batch_size", 1))
        seq_len = seq_len or int(getattr(self.config, "seq_len", 1))
        tensor = torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)
        self._set_Z_parameter(tensor)
        self.init_exponential_increase_filter(seq_len=seq_len)
        self._rebuild_Z_optimizer()
        return self.Z

    def _ensure_Z_shape(self, batch_size: int, seq_len: int) -> torch.Tensor:
        if self.Z.shape[0] == batch_size and self.Z.shape[1] == seq_len:
            return self.Z

        with torch.no_grad():
            tensor = torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)
            min_batch = min(batch_size, self.Z.shape[0])
            min_seq = min(seq_len, self.Z.shape[1])
            if min_batch > 0 and min_seq > 0:
                tensor[:min_batch, :min_seq] = self.Z[:min_batch, :min_seq]
        self._set_Z_parameter(tensor)
        self.init_exponential_increase_filter(seq_len=seq_len)
        self._rebuild_Z_optimizer()
        return self.Z

    def reset_Z(
        self,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> None:
        batch_size = batch_size or int(getattr(self.config, "batch_size", 1))
        seq_len = seq_len or int(getattr(self.config, "seq_len", 1))
        if self.Z.shape == (batch_size, seq_len, self.Z_dim):
            with torch.no_grad():
                self.Z.zero_()
            self.init_exponential_increase_filter(seq_len=seq_len)
            self.Z.requires_grad_(True)
            self.latent = self.Z
            return

        tensor = torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)
        self._set_Z_parameter(tensor)
        self.init_exponential_increase_filter(seq_len=seq_len)
        self._rebuild_Z_optimizer()

    def detach_Z(self) -> None:
        self.Z.detach_()
        self.Z.requires_grad_(True)
        self.latent = self.Z

    def set_Z(self, Z: torch.Tensor) -> None:
        tensor = Z.to(self.device)
        if tensor.shape == self.Z.shape:
            with torch.no_grad():
                self.Z.copy_(tensor)
            self.Z.requires_grad_(True)
            self.latent = self.Z
            self.init_exponential_increase_filter(seq_len=tensor.shape[1])
            return

        if not tensor.requires_grad:
            tensor = tensor.detach()
        self._set_Z_parameter(tensor)
        self.init_exponential_increase_filter(seq_len=tensor.shape[1])
        self._rebuild_Z_optimizer()

    def _set_Z_parameter(self, tensor: torch.Tensor) -> None:
        param = nn.Parameter(tensor, requires_grad=True)
        self.Z = param
        self.latent = self.Z  # legacy alias without registering twice

    def _rebuild_Z_optimizer(self) -> None:
        self.Z_optimizer = self._build_Z_optimizer()
        self.LU_optimizer = self.Z_optimizer

    # ------------------------------------------------------------------ #
    # Optimisers
    # ------------------------------------------------------------------ #

    def _build_W_optimizer(self):
        params = [p for name, p in self.named_parameters() if name not in {"Z"}]
        if not params:
            return None

        lr = float(getattr(self.config, "WU_lr", 1e-3))
        weight_decay = float(getattr(self.config, "l2_loss", 0.0) or 0.0)
        opt_name = getattr(self.config, "WU_optimizer", "Adam").lower()

        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if opt_name == "sgd":
            momentum = float(getattr(self.config, "WU_momentum", 0.0))
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"Unsupported WU_optimizer '{self.config.WU_optimizer}'.")

    def _optimizer_wrapper(self):
        if self.Z_chunks <= 1:
            return self.Z_optimizer

    def _build_Z_optimizer(self):
        lr = float(getattr(self.config, "LU_lr", 1e-1))
        weight_decay = 0.0  # handled manually per chunk
        opt_name = getattr(self.config, "LU_optimizer", "Adam").lower()

        if opt_name == "adam":
            betas = getattr(self.config, "LU_Adam_betas", (0.9, 0.999))
            return torch.optim.Adam([self.Z], lr=lr, weight_decay=weight_decay, betas=betas)
        if opt_name == "adamw": # quick experiment to see if it matters. AdamW impleements weight decay separately from adaptive LR.
            betas = getattr(self.config, "LU_Adam_betas", (0.9, 0.999))
            return torch.optim.AdamW([self.Z], lr=lr, weight_decay=weight_decay, betas=betas)
        if opt_name == "sgd":
            momentum = float(getattr(self.config, "LU_momentum", 0.0))
            return torch.optim.SGD([self.Z], lr=lr, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"Unsupported LU_optimizer '{self.config.LU_optimizer}'.")

    # ------------------------------------------------------------------ #
    # Latent helpers
    # ------------------------------------------------------------------ #

    def latent_activation_function(self, x: torch.Tensor) -> torch.Tensor:
        activation = getattr(self.config, "latent_activation", "softmax")
        if activation == "softmax":
            temperature = float(getattr(self.config, "softmax_temp", 1.0))
            return torch.softmax(x / temperature, dim=-1)
        if activation == "softmax_chunked":
            chunk_size = self.Z_dim // self.Z_chunks
            chunks = torch.split(x, chunk_size, dim=-1)
            activated = [torch.softmax(chunk, dim=-1) for chunk in chunks]
            return torch.cat(activated, dim=-1)
        if activation == "sigmoid":
            return torch.sigmoid(x)
        if activation == "none":
            return x
        raise ValueError(f"Unsupported latent_activation '{activation}'.")

    def _prepare_latent_tensor(
        self,
        batch_size: int,
        seq_len: int,
        what_latent: str,
        taskID: Optional[torch.Tensor],
    ) -> torch.Tensor:
        what_latent = what_latent or "self"
        if what_latent == "self":
            return self._ensure_Z_shape(batch_size, seq_len)

        if what_latent in {"uniform", "init"}:
            value = 1.0 / self.Z_dim
            return torch.full((batch_size, seq_len, self.Z_dim), value, device=self.device)

        if what_latent == "zeros":
            return torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)

        if what_latent == "taskID":
            if taskID is None:
                raise ValueError("taskID must be provided when what_latent='taskID'.")
            ids = taskID.to(self.device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(1).expand(-1, seq_len)
            latent = torch.zeros(batch_size, seq_len, self.Z_dim, device=self.device)
            latent.scatter_(dim=2, index=ids.long().unsqueeze(-1), value=1.0)
            return latent

        raise ValueError(f"Unsupported latent selection '{what_latent}'.")

    def _get_Z_slice(
        self,
        seq_step: int,
        batch_size: int,
        what_latent: str,
        taskID: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seq_len = max(1, self.Z.shape[1])
        seq_step = min(seq_step, seq_len - 1)

        if what_latent == "self":
            return self.Z[:, seq_step, :]

        if what_latent in {"uniform", "init"}:
            value = 1.0 / self.Z_dim
            return torch.full((batch_size, self.Z_dim), value, device=self.device)

        if what_latent == "zeros":
            return torch.zeros(batch_size, self.Z_dim, device=self.device)

        if what_latent == "taskID":
            if taskID is None:
                raise ValueError("taskID must be provided when what_latent='taskID'.")
            ids = taskID.to(self.device)
            if ids.dim() == 1:
                ids = ids.long()
            else:
                ids = ids[:, seq_step].long()
            latent = torch.zeros(batch_size, self.Z_dim, device=self.device)
            latent.scatter_(dim=1, index=ids.unsqueeze(-1), value=1.0)
            return latent

        raise ValueError(f"Unsupported latent selection '{what_latent}'.")

    # ------------------------------------------------------------------ #
    # Input preparation utilities
    # ------------------------------------------------------------------ #

    def _prepend_zero_frame(self, input: torch.Tensor) -> torch.Tensor:
        zero_frame = torch.zeros_like(input[:, :1, :])
        return torch.cat((zero_frame, input[:, :-1, :]), dim=1)

    def _prepare_inputs_for_forward(self, input: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "predict_first_frame", False):
            return self._prepend_zero_frame(input)
        return input

    def combine_input_with_latent(
        self,
        input: torch.Tensor,
        what_latent: str = "self",
        taskID: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = input.shape
        latent = self._prepare_latent_tensor(batch_size, seq_len, what_latent, taskID)
        latent = self.latent_activation_function(latent)

        processed_input = (
            self._prepend_zero_frame(input)
            if getattr(self.config, "predict_first_frame", False)
            else input
        )
        return torch.cat((processed_input, latent), dim=-1)

    # ------------------------------------------------------------------ #
    # Multiplicative gating
    # ------------------------------------------------------------------ #

    def _select_gate(self, stage: str) -> torch.Tensor:
        if stage == "pre":
            mask = self.gating_mask1
        elif stage == "post":
            mask = self.gating_mask2
        else:
            raise ValueError(f"Unknown gating stage '{stage}'.")

        if mask is None:
            raise RuntimeError("Multiplicative gating requested but masks were not initialised.")
        return mask

    def _project_gates(self, latent_slice: torch.Tensor, stage: str) -> torch.Tensor:
        mask = self._select_gate(stage)
        latent_slice = self.latent_activation_function(latent_slice)
        return torch.matmul(latent_slice, mask)

    def apply_mul_gating(
        self,
        hidden_state: torch.Tensor,
        cell_state: Optional[torch.Tensor],
        seq_step: int,
        stage: str = "pre",
        what_latent: str = "self",
        taskID: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = hidden_state.shape[0]
        self._ensure_Z_shape(batch_size, max(self.Z.shape[1], seq_step + 1))
        latent_slice = self._get_Z_slice(seq_step, batch_size, what_latent, taskID)

        gates = self._project_gates(latent_slice, stage)
        hidden_state = hidden_state * gates
        if cell_state is not None:
            cell_state = cell_state * gates
        return hidden_state, cell_state

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input: torch.Tensor,
        taskID: Optional[torch.Tensor] = None,
        what_latent: str = "self",
    ):
        batch_size, seq_len, _ = input.shape
        self.init_hidden(batch_size)

        processed_input = self.input_layer(input)
        outputs: List[torch.Tensor] = []
        for step in range(seq_len):
            if self.use_mul_gating and getattr(self.config, "pre_gating", False):
                self.hidden_state, self.cell_state = self.apply_mul_gating(
                    self.hidden_state,
                    self.cell_state,
                    step,
                    stage="pre",
                    what_latent=what_latent,
                    taskID=taskID,
                )

            if self.config.rnn_type.lower() == "lstm":
                self.hidden_state, self.cell_state = self.rnn_cell(
                    processed_input[:, step, :], (self.hidden_state, self.cell_state)
                )
            else:
                self.hidden_state = self.rnn_cell(processed_input[:, step, :], self.hidden_state)

            if self.use_mul_gating and getattr(self.config, "post_gating", False):
                self.hidden_state, self.cell_state = self.apply_mul_gating(
                    self.hidden_state,
                    self.cell_state,
                    step,
                    stage="post",
                    what_latent=what_latent,
                    taskID=taskID,
                )

            outputs.append(self.output_layer(self.hidden_state))

        return outputs, (self.hidden_state, self.cell_state)

    # ------------------------------------------------------------------ #
    # Latent (Z) optimisation
    # ------------------------------------------------------------------ #

    def update_Z(
        self,
        input: torch.Tensor,
        loss_function: Optional[nn.Module] = None,
        logger=None,
        taskID: Optional[torch.Tensor] = None,
        no_of_latent_steps: Optional[int] = None,
    ):
        steps = (
            int(getattr(self.config, "no_of_steps_in_latent_space", 0))
            if no_of_latent_steps is None
            else int(no_of_latent_steps)
        )
        if steps <= 0:
            return None

        loss_function = loss_function or self.loss_function
        before_optim_loss = None

        for step in range(steps):
            self.Z_optimizer.zero_grad()

            if self.use_add_gating:
                model_input = self.combine_input_with_latent(input, what_latent="self", taskID=taskID)
            else:
                model_input = self._prepare_inputs_for_forward(input)

            outputs, _ = self.forward(model_input, taskID=taskID, what_latent="self")
            outputs = torch.stack(outputs, dim=1)

            if getattr(self.config, "predict_first_frame", False):
                target = input
            else:
                target = input[:, 1:, :]
            target = target.to(outputs.device)
            if outputs.shape[1] != target.shape[1]:
                target = target[:, : outputs.shape[1], :]

            loss = loss_function(outputs, target)
            if before_optim_loss is None:
                before_optim_loss = loss.detach().cpu().numpy()

            reduction = getattr(self.config, "loss_reduction_LU", "sum")
            loss_scalar = loss.sum() if reduction == "sum" else loss.mean()
            loss_scalar.backward()

            self.adjust_Z_grads(getattr(self.config, "latent_aggregation_op", "average"))
            self.Z_optimizer.step()

            if logger is not None:
                if hasattr(logger, "log_updating_loss"):
                    logger.log_updating_loss(loss.detach().cpu().numpy())
                if hasattr(logger, "log_updating_latent"):
                    logger.log_updating_latent(self.Z.detach().cpu().numpy())
                if hasattr(logger, "log_updating_output"):
                    logger.log_updating_output(outputs.detach().cpu().numpy())

        return before_optim_loss

    def adjust_Z_grads(self, apply_op: str = "average") -> None:
        if self.Z.grad is None:
            return

        grads = self.Z.grad
        if apply_op == "average":
            mean = grads.mean(dim=1, keepdim=True)
            self.Z.grad = mean.expand_as(grads).clone()
        elif apply_op == "exponential":
            weights = torch.exp(torch.arange(grads.shape[1], device=grads.device))
            weights = weights / weights.sum()
            weighted = (grads * weights.view(1, -1, 1)).sum(dim=1, keepdim=True)
            self.Z.grad = weighted.expand_as(grads).clone()
        elif apply_op == "last":
            last = grads[:, -1:, :]
            self.Z.grad = last.expand_as(grads).clone()
        elif apply_op == "mask_last":
            grads = grads.clone()
            grads[:, -1, :] = 0.0
            mean = grads.mean(dim=1, keepdim=True)
            self.Z.grad = mean.expand_as(grads).clone()
        elif apply_op == "mask_all":
            self.Z.grad.zero_()
        elif apply_op == "exponential_increase":
            self._apply_exponential_increase()
        elif apply_op == "none":
            pass
        else:
            raise ValueError(f"Unsupported latent_aggregation_op '{apply_op}'.")

        self._apply_chunk_lr_and_decay()

    def init_exponential_increase_filter(
        self,
        config_or_seq_len: Optional[Union[int, object]] = None,
        seq_len: Optional[int] = None,
    ) -> None:
        if isinstance(config_or_seq_len, int):
            seq_len = config_or_seq_len
        elif hasattr(config_or_seq_len, "seq_len"):
            seq_len = getattr(config_or_seq_len, "seq_len")

        seq_len = seq_len or int(getattr(self.config, "seq_len", 0))
        if seq_len <= 0:
            self.exponential_increase_filter = None
            return

        x = torch.linspace(0, seq_len - 1, steps=seq_len, device=self.device)
        steepness_values = getattr(self.config, "exponential_increase_steepness", [0.0])
        multipliers = getattr(self.config, "exponential_increase_multipliers", [1.0])

        filters: List[torch.Tensor] = []
        for idx in range(self.Z_chunks):
            steepness = float(
                steepness_values[idx] if idx < len(steepness_values) else steepness_values[-1]
            )
            multiplier = float(
                multipliers[idx] if idx < len(multipliers) else multipliers[-1]
            )
            initial_value = torch.exp(torch.tensor(-steepness, device=self.device))
            rate = steepness / max(seq_len, 1)
            y = initial_value * torch.exp(rate * x)
            y = y / y.sum()
            filters.append(y * multiplier)

        self.exponential_increase_filter = filters[0] if self.Z_chunks == 1 else filters

    def _apply_exponential_increase(self) -> None:
        if self.Z.grad is None or self.exponential_increase_filter is None:
            return

        grads = self.Z.grad.clone()
        if isinstance(self.exponential_increase_filter, list):
            chunk_size = self.Z_dim // self.Z_chunks
            for idx, filt in enumerate(self.exponential_increase_filter):
                filt = filt.to(grads.device)
                grad_chunk = grads[:, :, idx * chunk_size : (idx + 1) * chunk_size]
                if grad_chunk.numel() == 0:
                    continue
                grad_chunk.mul_(filt.view(1, -1, 1))
                grads[:, :, idx * chunk_size : (idx + 1) * chunk_size] = grad_chunk
        else:
            filt = self.exponential_increase_filter.to(grads.device)
            grads.mul_(filt.view(1, -1, 1))

        mean = grads.mean(dim=1, keepdim=True)
        self.Z.grad = mean.expand_as(self.Z.grad).clone()

    def _apply_chunk_lr_and_decay(self) -> None:
        '''
        Apply different learning rates and weight decays to different chunks of the latent variable Z. Used in creating multiple Z populations with different dynamics.
        '''
        if self.Z.grad is None:
            return

        grad = self.Z.grad
        base_lr = float(getattr(self.config, "LU_lr", 1.0)) or 1.0
        chunk_lrs = getattr(self.config, "chunk_LU_lrs", None)
        base_decay = float(getattr(self.config, "l2_loss", 0.0) or 0.0)
        chunk_decays = getattr(self.config, "chunk_l2_losses", None)
        chunk_size = self.Z_dim // self.Z_chunks

        for idx in range(self.Z_chunks):
            start = idx * chunk_size
            end = min(self.Z_dim, (idx + 1) * chunk_size)
            if start >= end:
                continue
            grad_slice = grad[:, :, start:end]
            z_slice = self.Z[:, :, start:end]

            lr_value = None
            if chunk_lrs and idx < len(chunk_lrs):
                lr_value = float(chunk_lrs[idx])
            lr_scale = lr_value / base_lr if lr_value is not None and base_lr != 0 else 1.0
            if lr_scale != 1.0:
                grad_slice.mul_(lr_scale)

            decay_value = None
            if chunk_decays and idx < len(chunk_decays):
                decay_value = float(chunk_decays[idx])
            decay = decay_value if decay_value is not None else base_decay
            if decay:
                grad_slice.add_(decay * z_slice)

    # ------------------------------------------------------------------ #
    # RL helpers
    # ------------------------------------------------------------------ #
    def pick_action(
        self,
        action_distribution: torch.Tensor,
        offline_action: Optional[torch.Tensor] = None,
    ):
        categorical = torch.distributions.Categorical(action_distribution)
        action = offline_action if offline_action is not None else categorical.sample()
        log_prob = categorical.log_prob(action)
        return action, log_prob

    # ------------------------------------------------------------------ #
    # Legacy method aliases (keep external API stable)
    # ------------------------------------------------------------------ #

    def init_latent(self, *args, **kwargs):
        return self.init_Z(*args, **kwargs)

    def reset_latent(self, *args, **kwargs):
        return self.reset_Z(*args, **kwargs)

    def detach_latent(self):
        return self.detach_Z()

    def set_latent(self, *args, **kwargs):
        return self.set_Z(*args, **kwargs)

    def get_WU_optimizer(self):
        return self.W_optimizer

    def get_LU_optimizer(self):
        return self.Z_optimizer

    def update_latent(self, *args, **kwargs):
        return self.update_Z(*args, **kwargs)

    def adjust_latent_grads(self, *args, **kwargs):
        return self.adjust_Z_grads(*args, **kwargs)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        new_config = copy.deepcopy(self.config, memo)
        new_model = self.__class__(new_config).to(self.device)

        state = {k: v.clone() for k, v in self.state_dict().items()}
        new_model.load_state_dict(state, strict=True)

        memo[id(self)] = new_model
        return new_model
