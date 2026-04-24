"""
T-LoRA: Timestep-aware Low-Rank Adaptation for diffusion models.

Built on top of Hugging Face PEFT, T-LoRA dynamically adjusts the effective
LoRA rank based on the diffusion timestep.  Higher timesteps (more noise) use
fewer rank components, reducing overfitting while preserving expressiveness at
low-noise timesteps.

Quick start::

    from peft import LoraConfig
    from tlora_peft import get_tlora_peft_model, set_timestep

    config = LoraConfig(r=32, lora_alpha=32, target_modules=[...])

    # Standard T-LoRA
    model = get_tlora_peft_model(model, config, min_rank=1, max_timestep=1000)

    # Orthogonal T-LoRA
    model = get_tlora_peft_model(model, config, min_rank=1, ortho=True,
                                  sig_type="last", ortho_init="random")

    # In the denoising loop:
    set_timestep(model, current_timestep)
    output = model(noisy_input, timestep=t, ...)
"""

from .layer import TLoRALinear, TLoRAOrthoLinear
from .utils import (
    get_tlora_peft_model,
    add_tlora_adapter,
    set_timestep,
    enable_timestep_hook,
    save_tlora_adapter,
    save_tlora_metadata,
    load_tlora_weights,
    load_tlora_metadata,
    load_adapter,
    set_adapter,
    enable_adapter,
    disable_adapter,
    remove_adapter,
    list_adapters,
    cast_tlora_adapter_dtype,
)

__all__ = [
    "TLoRALinear",
    "TLoRAOrthoLinear",
    "get_tlora_peft_model",
    "add_tlora_adapter",
    "set_timestep",
    "enable_timestep_hook",
    "save_tlora_adapter",
    "save_tlora_metadata",
    "load_tlora_weights",
    "load_tlora_metadata",
    "load_adapter",
    "set_adapter",
    "enable_adapter",
    "disable_adapter",
    "remove_adapter",
    "list_adapters",
    "cast_tlora_adapter_dtype",
]
