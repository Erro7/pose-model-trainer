from __future__ import annotations

import ctypes
import gc

import torch
import torch.nn.functional as F
from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.core.offload_training import manager as offload_manager
from diffsynth.core.offload_training import memory_buffer
from diffsynth.models import minimax_h3_image_dit
from diffsynth.models.model_loader import ModelPool
from entropack.linear.linear import CompressedLinear


_INSTALLED = False


def _is_compressed(value) -> bool:
    return value is not None and hasattr(value, "entropack_quant_state")


class _PlaceholderOffloader:
    def onload(self) -> None:
        pass

    def offload(self) -> None:
        pass

    def offload_grad(self) -> None:
        pass


class _CompressedParamOffloader:
    def __init__(self, module: torch.nn.Module, name: str, target_device) -> None:
        self.module = module
        self.name = name
        self.target_device = target_device
        param = module._parameters[name]
        self.dtype = param.dtype
        self.cpu_copy = param.detach().cpu()
        self.offload()

    def onload(self) -> None:
        moved = self.cpu_copy.to(self.target_device, non_blocking=True)
        self.module._parameters[self.name] = torch.nn.Parameter(moved, requires_grad=False)

    def offload(self) -> None:
        self.module._parameters[self.name] = torch.nn.Parameter(
            torch.empty(0, device=self.target_device, dtype=self.dtype),
            requires_grad=False,
        )

    def offload_grad(self) -> None:
        pass


class _CompressedBufferOffloader:
    def __init__(self, module: torch.nn.Module, name: str, buffer, target_device) -> None:
        self.module = module
        self.name = name
        self.target_device = target_device
        self.cpu_copy = buffer.detach().cpu()

    def onload(self) -> None:
        self.module._buffers[self.name] = self.cpu_copy.to(self.target_device, non_blocking=True)

    def offload(self) -> None:
        self.module._buffers[self.name] = self.cpu_copy

    def offload_grad(self) -> None:
        pass


def _parameter_slots(model: torch.nn.Module, params):
    wanted = None if params is None else {id(param) for param in params}
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if wanted is None or id(param) in wanted:
                yield module, name, param


def _unit_init(
    self,
    model,
    target_device,
    enable_optimizer_cpu_offload=False,
    params=None,
    buffers=None,
    memory_buffer=None,
):
    self.model = model
    self.target_device = target_device
    self.param_offloaders = {}
    self.compressed_by_module = {}
    for module, name, param in _parameter_slots(model, params):
        if not param.requires_grad:
            if _is_compressed(param.data):
                offloader = _CompressedParamOffloader(module, name, target_device)
                self.compressed_by_module.setdefault(id(module), []).append(offloader)
                self.param_offloaders[id(module._parameters[name])] = _PlaceholderOffloader()
            else:
                self.param_offloaders[id(param)] = offload_manager.StaticParamOffloader(
                    param,
                    target_device,
                    memory_buffer=memory_buffer,
                )
        elif enable_optimizer_cpu_offload:
            self.param_offloaders[id(param)] = offload_manager.TrainableParamOffloader(param, target_device)
        else:
            self.param_offloaders[id(param)] = offload_manager.AlwaysOnGPUParamOffloader(param, target_device)

    for module, name, buf in buffers or []:
        if _is_compressed(buf):
            offloader = _CompressedBufferOffloader(module, name, buf, target_device)
            self.compressed_by_module.setdefault(id(module), []).append(offloader)
            self.param_offloaders[id(buf)] = _PlaceholderOffloader()
        else:
            self.param_offloaders[id(buf)] = offload_manager.BufferOffloader(
                module,
                name,
                buf,
                target_device,
                memory_buffer=memory_buffer,
            )


def _onload_module(self, module):
    for offloader in self.compressed_by_module.get(id(module), ()):
        offloader.onload()
    _ORIGINAL_ONLOAD_MODULE(self, module)


def _offload_module(self, module):
    _ORIGINAL_OFFLOAD_MODULE(self, module)
    for offloader in self.compressed_by_module.get(id(module), ()):
        offloader.offload()


_ORIGINAL_ONLOAD_MODULE = offload_manager.UnitWiseParamManager.onload_module
_ORIGINAL_OFFLOAD_MODULE = offload_manager.UnitWiseParamManager.offload_module


def _pool_from_model(cls, model, min_chunk_size=1 * 1024**3, max_chunk_size=4 * 1024**3):
    tensors = [param for param in model.parameters() if not param.requires_grad and not _is_compressed(param.data)]
    tensors += [buffer for buffer in model.buffers() if not _is_compressed(buffer)]
    total = sum(memory_buffer._tensor_storage_size(tensor) for tensor in tensors)
    max_tensor_size = max((memory_buffer._tensor_storage_size(tensor) for tensor in tensors), default=0)
    if max_tensor_size > max_chunk_size:
        max_chunk_size = memory_buffer._next_power_of_two(max_tensor_size)
    return cls(total, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)


class _CompressedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer):
        ctx.layer = layer
        ctx.input_shape = x.shape
        weight = layer.dequantize(x.device).to(x.dtype)
        return F.linear(x, weight, layer._bias_on(x.device))

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.layer.dequantize(grad_output.device).to(grad_output.dtype)
        grad_input = grad_output.reshape(-1, grad_output.shape[-1]).matmul(weight)
        return grad_input.reshape(ctx.input_shape), None


_ORIGINAL_COMPRESSED_FORWARD = CompressedLinear.forward


def _compressed_linear_forward(self, x):
    if _is_compressed(self.weight):
        return _CompressedLinearFunction.apply(x, self)
    return _ORIGINAL_COMPRESSED_FORWARD(self, x)


def _pruned_forward_with_checkpoint(
    self,
    x,
    prompt_embeds,
    position_ids,
    unique_timesteps,
    timestep_indices,
    token_tags,
    img_pos,
    text_pos,
    pdd_heads=None,
    pdd_step_index=0,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if x.dim() != 2:
        raise ValueError(f"x must be [N_img, patch_dim], got {list(x.shape)}")
    if pdd_heads is None:
        pdd_heads = getattr(self, "pdd_heads", None)
    if pdd_heads is None:
        raise ValueError("MiniMaxH3ImageDiTPruned has no usable output head: pass `pdd_heads`.")

    device = x.device
    seq_len = int(position_ids.shape[1])
    img_pos = img_pos.view(-1).to(device=device, dtype=torch.long)
    text_pos = text_pos.view(-1).to(device=device, dtype=torch.long)
    timestep_indices = timestep_indices.view(-1).to(device=device, dtype=torch.long)
    token_tags = token_tags.view(-1).to(device=device, dtype=torch.long)

    rope_freqs = self.rope(position_ids.to(device))
    hidden = self._embed(
        x=x,
        prompt_embeds=prompt_embeds,
        seq_len=seq_len,
        img_pos=img_pos,
        text_pos=text_pos,
        device=device,
    )
    timestep_rows = torch.full(
        (unique_timesteps.shape[0],),
        int(pdd_step_index),
        dtype=torch.long,
        device=device,
    )
    timestep_rows[1:] = minimax_h3_image_dit.MINIMAX_H3_IMAGE_ADALN_TABLE_ROWS - 1
    combined_indices = (
        timestep_indices * minimax_h3_image_dit.MINIMAX_H3_ADALN_MODALITY_NUM
        + token_tags.clamp(min=0)
    ).to(device)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    for block in self.blocks:
        hidden = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            hidden,
            t_emb=timestep_rows,
            combined_indices=combined_indices,
            rope_freqs=rope_freqs,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
        )

    hidden = self.final_layer(hidden, t_emb=timestep_rows, timestep_indices=timestep_indices)
    output = pdd_heads(hidden.to(pdd_heads.dtype), pdd_step_index)
    return output.index_select(0, img_pos)


_ORIGINAL_AUTO_LOAD = ModelPool.auto_load_model


def _auto_load_with_trim(self, *args, **kwargs):
    result = _ORIGINAL_AUTO_LOAD(self, *args, **kwargs)
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    offload_manager.UnitWiseParamManager.__init__ = _unit_init
    offload_manager.UnitWiseParamManager.onload_module = _onload_module
    offload_manager.UnitWiseParamManager.offload_module = _offload_module
    offload_manager.PinnedArenaPool.from_model = classmethod(_pool_from_model)
    CompressedLinear.forward = _compressed_linear_forward
    minimax_h3_image_dit.MiniMaxH3ImageDiTPruned.forward = _pruned_forward_with_checkpoint
    ModelPool.auto_load_model = _auto_load_with_trim
    _INSTALLED = True
