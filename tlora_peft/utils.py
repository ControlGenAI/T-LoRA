"""
Utility functions for applying T-LoRA to any model, managing timesteps,
and saving/loading adapters with T-LoRA metadata.
"""

import gc
import json
import os
import re
import warnings
from typing import Optional

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_peft_model_state_dict

from .layer import TLoRALinear, TLoRAOrthoLinear

_TLORA_LAYERS = (TLoRALinear, TLoRAOrthoLinear)
_PEFT_PREFIXES = ("base_model.model.", "base_model.")


def _strip_peft_prefix(key):
    """Strip PEFT wrapper prefixes to get the canonical module path."""
    for pfx in _PEFT_PREFIXES:
        if key.startswith(pfx):
            return key[len(pfx):]
    return key


def _normalize_lora_key(key):
    """Strip PEFT prefix AND adapter name to get a universal canonical key.

    Model params:
        base_model.model.xxx.lora_A.default.weight
        base_model.model.xxx.lora_lambda.default   (ParameterDict, no ``.weight``)
    State dict (PEFT may or may not strip ``.<adapter_name>``):
        xxx.lora_A.weight   or   xxx.lora_A.default.weight
        xxx.lora_lambda     or   xxx.lora_lambda.default
    Both become:
        xxx.lora_A.weight
        xxx.lora_lambda
    """
    k = _strip_peft_prefix(key)
    # ``lora_A`` / ``lora_B`` — ModuleDict of Linear, keys end in ``.weight``
    k = re.sub(r"\.lora_([AB])\.[^.]+\.weight$", r".lora_\1.weight", k)
    # ``lora_lambda`` — ParameterDict, keys end in ``.<adapter_name>``
    k = re.sub(r"\.lora_lambda\.[^.]+$", ".lora_lambda", k)
    return k


def _normalize_frozen_key(key):
    """Canonical form for ``tlora_frozen.pt`` keys — prefix- and adapter-name agnostic.

    Handles every combination of wrapping (PeftModel vs raw) and adapter
    naming (``default`` vs ``default_0`` etc.) so that a frozen file saved
    one way can be loaded the other way.
    """
    k = _strip_peft_prefix(key)
    k = re.sub(r"\.base_([AB])\.[^.]+\.weight$", r".base_\1.weight", k)
    k = re.sub(r"\.base_lambda\.[^.]+$", ".base_lambda", k)
    return k


def _load_weights_by_canonical_name(model, state_dict, adapter_name="default"):
    """Load adapter weights by canonical name, prefix and adapter-name agnostic.

    Returns:
        ``(loaded, n_sd, missing_model_params)``:
          * ``loaded``               — state-dict keys written into the model
          * ``n_sd``                 — total state-dict keys (``lora_*`` ones)
          * ``missing_model_params`` — model ``lora_*`` params for *adapter_name*
                                       that never received a matching key — if
                                       non-empty, those tensors keep their
                                       random init (and will differ between
                                       runs, hurting reproducibility).
    """
    param_map = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        # Only check params belonging to the adapter we are loading — other
        # adapters' params legitimately won't appear in this state dict.
        if f".{adapter_name}." not in name and not name.endswith(f".{adapter_name}"):
            continue
        param_map[_normalize_lora_key(name)] = (name, param)

    loaded_canons = set()
    n_sd = 0
    for sd_key, sd_val in state_dict.items():
        if "lora_" not in sd_key:
            continue
        n_sd += 1
        canon = _normalize_lora_key(sd_key)
        if canon in param_map:
            _, p = param_map[canon]
            p.data.copy_(sd_val.to(device=p.device, dtype=p.dtype))
            loaded_canons.add(canon)

    missing = [full for canon, (full, _) in param_map.items() if canon not in loaded_canons]
    return len(loaded_canons), n_sd, missing


def get_tlora_peft_model(
    model: nn.Module,
    lora_config: LoraConfig,
    min_rank: int = 1,
    max_timestep: int = 1000,
    ortho: bool = False,
    sig_type: str = "last",
    ortho_init: str = "random",
    adapter_name: str = "default",
    skip_ortho_init: bool = False,
):
    """Wrap *model* with PEFT LoRA using T-LoRA layers.

    Analogous to ``peft.get_peft_model``, but registers a T-LoRA layer class
    as a custom PEFT module so that every ``nn.Linear`` matched by
    ``lora_config.target_modules`` gets a timestep-aware adapter.

    Args:
        model: The base model (transformer, UNet, DiT, ...).
        lora_config: A standard ``peft.LoraConfig``.  Tip: set
            ``lora_alpha = r`` for effective scaling = 1.0 (T-LoRA default).
        min_rank: Minimum effective rank at the highest timestep.
        max_timestep: The maximum timestep value in the diffusion schedule
            (1000 for most schedulers).
        ortho: If ``True``, use :class:`TLoRAOrthoLinear` (SVD-initialised
            Q/P/λ) instead of :class:`TLoRALinear`.
        sig_type: *(ortho only)* Which singular components to keep —
            ``"last"`` (default, smallest), ``"principal"`` (largest),
            ``"middle"``.
        ortho_init: *(ortho only)* SVD source — ``"random"`` (default) or
            ``"base_layer"`` (SVD of the original weight matrix).
        skip_ortho_init: *(ortho only)* If ``True``, skip the SVD
            initialisation and zero-initialise lora_A/B/λ + frozen copies.
            Use only when you will immediately overwrite the weights from a
            saved checkpoint — this is what :func:`load_tlora_weights`
            does internally to avoid wasting time on an SVD whose result is
            about to be thrown away.

    Returns:
        The model with T-LoRA layers (PeftModel wrapper or the original model
        with adapters injected in-place, depending on PEFT version).
    """
    if ortho:
        layer_cls = TLoRAOrthoLinear
        saved = (
            TLoRAOrthoLinear._default_sig_type,
            TLoRAOrthoLinear._default_ortho_init,
            TLoRAOrthoLinear._skip_init,
        )
        TLoRAOrthoLinear._default_sig_type = sig_type
        TLoRAOrthoLinear._default_ortho_init = ortho_init
        TLoRAOrthoLinear._skip_init = skip_ortho_init
    else:
        layer_cls = TLoRALinear

    lora_config._register_custom_module({nn.Linear: layer_cls})

    try:
        try:
            peft_model = get_peft_model(
                model,
                lora_config,
                adapter_name=adapter_name,
                autocast_adapter_dtype=False,
            )
        except ValueError as e:
            if "not found in the base model" not in str(e):
                raise
            # PEFT >=0.19 + diffusers: get_peft_model can't resolve
            # target_modules that work fine via model.add_adapter().
            # Fall back to the same injection path the training script uses.
            if hasattr(model, "add_adapter"):
                try:
                    model.add_adapter(lora_config, adapter_name=adapter_name)
                except TypeError:
                    # Older diffusers API without adapter_name kwarg
                    model.add_adapter(lora_config)
            else:
                from peft import inject_adapter_in_model
                inject_adapter_in_model(lora_config, model, adapter_name)
            peft_model = model
    finally:
        if ortho:
            (
                TLoRAOrthoLinear._default_sig_type,
                TLoRAOrthoLinear._default_ortho_init,
                TLoRAOrthoLinear._skip_init,
            ) = saved

    for module in peft_model.modules():
        if isinstance(module, _TLORA_LAYERS):
            module.max_timestep = max_timestep
            for an in module.lora_A.keys():
                module.min_rank[an] = min_rank

    if ortho and not skip_ortho_init:
        gc.collect()
        torch.cuda.empty_cache()

    return peft_model


def add_tlora_adapter(
    model: nn.Module,
    lora_config: LoraConfig,
    adapter_name: str = "default",
    min_rank: int = 1,
    max_timestep: int = 1000,
    ortho: bool = False,
    sig_type: str = "last",
    ortho_init: str = "random",
    skip_ortho_init: bool = False,
):
    """Add a T-LoRA adapter to an already PEFT-wrapped model.

    Analogous to ``model.add_adapter()``, but registers the appropriate
    T-LoRA layer class first.  Use when you already have a PeftModel and
    want to add a new T-LoRA adapter (e.g. for multi-adapter scenarios).

    Note:
        If the model already contains T-LoRA layers (from an earlier
        ``get_tlora_peft_model`` call), PEFT adds new adapter weights to
        those *existing* layers — it does not create new ones.  The ``ortho``
        parameter only affects newly-created layers.

    ``skip_ortho_init=True`` skips the expensive SVD initialisation — use
    only when you are about to overwrite the weights from a checkpoint.
    """
    if ortho:
        layer_cls = TLoRAOrthoLinear
        saved = (
            TLoRAOrthoLinear._default_sig_type,
            TLoRAOrthoLinear._default_ortho_init,
            TLoRAOrthoLinear._skip_init,
        )
        TLoRAOrthoLinear._default_sig_type = sig_type
        TLoRAOrthoLinear._default_ortho_init = ortho_init
        TLoRAOrthoLinear._skip_init = skip_ortho_init
    else:
        layer_cls = TLoRALinear

    lora_config._register_custom_module({nn.Linear: layer_cls})

    try:
        # ``model`` may be a raw transformer (diffusers PeftAdapterMixin) with
        # ``add_adapter(config, adapter_name=...)`` or a ``PeftModel`` with
        # ``add_adapter(adapter_name, peft_config)`` — signatures differ.
        # Fall through ``inject_adapter_in_model`` which is stable in both cases.
        from peft import inject_adapter_in_model
        inject_target = getattr(model, "base_model", None)
        inject_target = getattr(inject_target, "model", None) or model
        inject_adapter_in_model(lora_config, inject_target, adapter_name)
        # Mirror what PeftModel.add_adapter does so ``peft_config`` stays consistent.
        if hasattr(model, "peft_config") and isinstance(model.peft_config, dict):
            model.peft_config[adapter_name] = lora_config
    finally:
        if ortho:
            (
                TLoRAOrthoLinear._default_sig_type,
                TLoRAOrthoLinear._default_ortho_init,
                TLoRAOrthoLinear._skip_init,
            ) = saved

    for module in model.modules():
        if isinstance(module, _TLORA_LAYERS):
            module.max_timestep = max_timestep
            module.min_rank[adapter_name] = min_rank

    if ortho and not skip_ortho_init:
        gc.collect()
        torch.cuda.empty_cache()

    return model


def set_timestep(model: nn.Module, timestep: int):
    """Set the current diffusion timestep on every T-LoRA layer.

    Must be called **before** each forward pass during both training and
    inference so that the rank mask is computed correctly.

    Args:
        model: The PEFT-wrapped model (or the raw transformer/UNet).
        timestep: Integer timestep in ``[0, max_timestep]``.
    """
    for module in model.modules():
        if isinstance(module, _TLORA_LAYERS):
            module.current_timestep = timestep


def enable_timestep_hook(
    model: nn.Module,
    timestep_pos: int = 1,
    timestep_key: str = "timestep",
    timestep_scale: float = 1.0,
):
    """Register a forward pre-hook that calls :func:`set_timestep` automatically.

    This makes T-LoRA work with **any** diffusers pipeline out of the box —
    the hook extracts the timestep from the model's forward arguments and
    broadcasts it to all T-LoRA layers before each forward pass.

    Args:
        model: The transformer / UNet (the actual ``nn.Module``, **not** the
            ``PeftModel`` wrapper — pass the object that the pipeline calls
            ``.forward()`` on).
        timestep_pos: Positional index of the timestep argument in the
            model's ``forward`` signature.  Defaults to ``1`` (second arg),
            which is standard for both Flux transformers
            (``forward(hidden_states, timestep, ...)``)) and SD-XL UNets
            (``forward(sample, timestep, ...)``).
        timestep_key: Keyword name of the timestep argument, used as a
            fallback when the timestep is passed as a keyword argument.
        timestep_scale: Multiplier applied to the raw timestep value
            before passing it to :func:`set_timestep`.  Use ``1000.0``
            for Flux (whose pipeline passes ``timestep / 1000`` to the
            transformer) and ``1.0`` (default) for SDXL / most schedulers
            that use integer timesteps.

    Returns:
        The ``RemovableHook`` handle (call ``.remove()`` to unregister).

    Example — SDXL::

        hook = enable_timestep_hook(pipeline.unet)            # scale=1.0
        images = pipeline(prompt, ...)
        hook.remove()

    Example — Flux::

        hook = enable_timestep_hook(pipeline.transformer,
                                     timestep_scale=1000.0)   # t*1000
        images = pipeline(prompt, ...)
        hook.remove()
    """

    def _pre_hook(module, args, kwargs):
        t = None
        if len(args) > timestep_pos:
            t = args[timestep_pos]
        elif timestep_key in kwargs:
            t = kwargs[timestep_key]

        if t is not None:
            if isinstance(t, torch.Tensor):
                t = t.flatten()[0].item()
            set_timestep(module, int(t * timestep_scale))

    return model.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def save_tlora_metadata(model: nn.Module, path: str, adapter_name: str = "default"):
    """Save ``tlora_meta.json`` alongside the PEFT adapter directory.

    Call this *after* ``model.save_pretrained(path)`` to persist the
    ``min_rank`` and ``max_timestep`` values that PEFT does not store.

    Note:
        Reads metadata from the *first* T-LoRA layer found; all layers are
        assumed to share the same config.
    """
    for module in model.modules():
        if isinstance(module, _TLORA_LAYERS):
            meta = {
                "rank": int(module.r[adapter_name]),
                "min_rank": int(module.min_rank.get(adapter_name, 1)),
                "max_timestep": int(module.max_timestep),
                "ortho": isinstance(module, TLoRAOrthoLinear),
            }
            if isinstance(module, TLoRAOrthoLinear):
                meta["sig_type"] = module.sig_type
                meta["ortho_init"] = module.ortho_init
            with open(os.path.join(path, "tlora_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            return meta
    return None


def load_tlora_metadata(path: str) -> dict:
    """Load ``tlora_meta.json`` from an adapter directory."""
    meta_path = os.path.join(path, "tlora_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _save_ortho_frozen(model: nn.Module, path: str, adapter_name: str = "default"):
    """Save frozen SVD copies that PEFT's standard save omits.

    :class:`TLoRAOrthoLinear` maintains frozen ``base_A``, ``base_B``,
    ``base_lambda`` — the "subtract to start at zero" baseline.  PEFT's
    ``get_peft_model_state_dict`` only saves *trainable* parameters, so
    these frozen copies are lost.  This function saves them separately.

    Keys are written in a prefix- and adapter-name-agnostic form so the
    same file loads correctly regardless of PEFT wrapping or the target
    adapter name chosen at inference time.
    """
    frozen: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, TLoRAOrthoLinear) and adapter_name in module.base_A:
            canon = _strip_peft_prefix(name)
            frozen[f"{canon}.base_A.weight"] = (
                module.base_A[adapter_name].weight.data.cpu()
            )
            frozen[f"{canon}.base_B.weight"] = (
                module.base_B[adapter_name].weight.data.cpu()
            )
            frozen[f"{canon}.base_lambda"] = (
                module.base_lambda[adapter_name].data.cpu()
            )
    if frozen:
        torch.save(frozen, os.path.join(path, "tlora_frozen.pt"))


def _load_ortho_frozen(
    model: nn.Module, frozen_path: str, adapter_name: str = "default",
):
    """Restore frozen SVD copies saved by :func:`_save_ortho_frozen`.

    Robust to:
      * old-format keys that embed the saver's adapter name
        (``xxx.base_A.default.weight``);
      * PEFT wrapper prefix mismatches between save and load
        (``base_model.model.xxx.…`` vs ``xxx.…``);
      * loading into a different adapter slot than was used at save time
        (e.g. ``default`` → ``default_0`` for multi-concept LoRAShop).
    """
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=True)
    frozen_by_canon = {_normalize_frozen_key(k): v for k, v in frozen.items()}

    loaded = 0
    expected = 0
    for name, module in model.named_modules():
        if isinstance(module, TLoRAOrthoLinear) and adapter_name in module.base_A:
            canon = _strip_peft_prefix(name)
            dev = module.base_A[adapter_name].weight.device
            dtype = module.base_A[adapter_name].weight.dtype
            a_key = f"{canon}.base_A.weight"
            b_key = f"{canon}.base_B.weight"
            l_key = f"{canon}.base_lambda"
            expected += 3
            if a_key in frozen_by_canon:
                module.base_A[adapter_name].weight.data.copy_(
                    frozen_by_canon[a_key].to(device=dev, dtype=dtype)
                )
                loaded += 1
            if b_key in frozen_by_canon:
                module.base_B[adapter_name].weight.data.copy_(
                    frozen_by_canon[b_key].to(device=dev, dtype=dtype)
                )
                loaded += 1
            if l_key in frozen_by_canon:
                module.base_lambda[adapter_name].data.copy_(
                    frozen_by_canon[l_key].to(device=dev, dtype=dtype)
                )
                loaded += 1

    if expected and loaded < expected:
        warnings.warn(
            f"Restored {loaded}/{expected} frozen SVD tensors from {frozen_path} "
            f"(adapter_name={adapter_name!r}). Ortho baseline may be incomplete.",
            stacklevel=2,
        )


def save_tlora_adapter(
    model: nn.Module, path: str, adapter_name: str = "default",
):
    """Save a T-LoRA adapter in PEFT-native format with full metadata.

    Creates the following files in *path*:

    * ``adapter_model.safetensors`` — trainable adapter weights
    * ``adapter_config.json`` — LoRA hyper-parameters
    * ``tlora_meta.json`` — T-LoRA specific metadata (min_rank, etc.)
    * ``tlora_frozen.pt`` *(ortho only)* — frozen SVD baseline copies

    Use :func:`load_tlora_weights` to load the adapter for inference.
    """
    os.makedirs(path, exist_ok=True)

    state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    try:
        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(path, "adapter_model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(path, "adapter_model.bin"))

    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        cfg = model.peft_config[adapter_name].to_dict()
        cfg.pop("_custom_module_mapping", None)
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    save_tlora_metadata(model, path, adapter_name)
    _save_ortho_frozen(model, path, adapter_name)


def _load_adapter_state_dict(adapter_path: str):
    """Load adapter weights from safetensors or bin file."""
    sf = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(sf):
        from safetensors.torch import load_file
        return load_file(sf)
    return torch.load(
        os.path.join(adapter_path, "adapter_model.bin"),
        map_location="cpu", weights_only=True,
    )


def _infer_target_modules(state_dict: dict) -> list[str]:
    """Extract fully-qualified module paths from adapter state dict keys.

    PEFT >=0.19 may fail to match short suffixes like ``"attn.to_q"`` during
    injection.  Using full paths (e.g. ``"transformer_blocks.0.attn.to_q"``)
    guarantees exact-match resolution regardless of PEFT version.
    """
    modules = set()
    for key in state_dict:
        if ".lora_A." not in key:
            continue
        k = _strip_peft_prefix(key)
        idx = k.index(".lora_A.")
        modules.add(k[:idx])
    return sorted(modules)


def load_tlora_weights(
    base_model: nn.Module,
    adapter_path: str,
    adapter_name: str = "default",
):
    """Load a T-LoRA adapter saved by :func:`save_tlora_adapter`.

    Wraps *base_model* with T-LoRA layers, loads the trained adapter
    weights, and restores frozen SVD copies (for ortho).

    Args:
        base_model: The raw model (transformer, UNet, …) — **not** a PeftModel.
        adapter_path: Directory containing the files produced by
            :func:`save_tlora_adapter`.
        adapter_name: Name for the loaded adapter (default ``"default"``).

    Returns:
        The model with T-LoRA adapter loaded (PeftModel wrapper or the
        original model with adapters injected in-place).

    Example — Flux inference::

        pipe = FluxPipeline.from_pretrained(model_id, ...).to("cuda")
        pipe.transformer = load_tlora_weights(pipe.transformer, adapter_path)
        hook = enable_timestep_hook(pipe.transformer, timestep_scale=1000.0)
        image = pipe(prompt=..., generator=...).images[0]
        hook.remove()
    """
    meta = load_tlora_metadata(adapter_path)
    if not meta:
        raise FileNotFoundError(
            f"tlora_meta.json not found in {adapter_path}. "
            "Was the adapter saved with save_tlora_adapter()?"
        )

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    state_dict = _load_adapter_state_dict(adapter_path)

    # Use full module paths from saved weights so PEFT finds them
    # regardless of how it resolves short suffixes like "attn.to_q".
    target_modules = _infer_target_modules(state_dict)
    if not target_modules:
        target_modules = cfg.get("target_modules")

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=cfg.get("lora_dropout", 0),
    )

    peft_model = get_tlora_peft_model(
        base_model,
        lora_config,
        min_rank=meta.get("min_rank", 1),
        max_timestep=meta.get("max_timestep", 1000),
        ortho=meta.get("ortho", False),
        sig_type=meta.get("sig_type", "last"),
        ortho_init=meta.get("ortho_init", "random"),
        adapter_name=adapter_name,
        skip_ortho_init=True,
    )

    loaded, n_sd, missing = _load_weights_by_canonical_name(
        peft_model, state_dict, adapter_name
    )
    if loaded == 0:
        raise RuntimeError(
            f"No adapter weights loaded from {adapter_path}. "
            f"State dict has {len(state_dict)} keys, sample: "
            f"{next(iter(state_dict), '(empty)')}"
        )
    if loaded < n_sd:
        warnings.warn(
            f"Loaded {loaded}/{n_sd} adapter keys from {adapter_path} — some keys did not match the model.",
            stacklevel=2,
        )
    if missing:
        warnings.warn(
            f"{len(missing)} adapter params in the model were NOT overwritten by "
            f"{adapter_path} — they keep their random init and will hurt reproducibility. "
            f"First missing: {missing[:3]}",
            stacklevel=2,
        )

    frozen_path = os.path.join(adapter_path, "tlora_frozen.pt")
    if os.path.isfile(frozen_path):
        _load_ortho_frozen(peft_model, frozen_path, adapter_name)
    elif meta.get("ortho", False):
        warnings.warn(
            f"tlora_frozen.pt not found in {adapter_path} but adapter is ortho. "
            "Frozen SVD copies will be from a fresh random init — results may be incorrect.",
            stacklevel=2,
        )

    return peft_model


def load_adapter(
    model: nn.Module,
    path: str,
    adapter_name: str,
    min_rank: Optional[int] = None,
    autocast_adapter_dtype: bool = False,
):
    """Load an extra T-LoRA adapter onto an already-wrapped T-LoRA model.

    ``path`` should contain ``adapter_model.safetensors`` (or ``.bin``),
    ``adapter_config.json`` and ``tlora_meta.json`` — everything produced by
    :func:`save_tlora_adapter`.

    Unlike the plain ``PeftModel.load_adapter`` path (which may fail to
    resolve short suffix ``target_modules`` like ``"attn.to_q"`` on some
    diffusers/PEFT combos and silently add the adapter to fewer modules
    than the first one), this helper mirrors :func:`load_tlora_weights`:

    1. Read the state dict and infer **fully-qualified** module paths.
    2. Inject the new adapter via :func:`add_tlora_adapter` with those full
       paths, so the second adapter targets exactly the same layers as the
       first.
    3. Copy weights through :func:`_load_weights_by_canonical_name`, which
       is prefix- and adapter-name-agnostic (handles saved keys without
       ``.default.`` as well as model keys with ``.default_N.``).
    4. Restore frozen SVD copies (for ortho variants).

    ``autocast_adapter_dtype`` is kept for API parity but unused —
    weights are cast to each target parameter's dtype at copy time.
    """
    meta = load_tlora_metadata(path)

    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    state_dict = _load_adapter_state_dict(path)

    target_modules = _infer_target_modules(state_dict)
    if not target_modules:
        target_modules = cfg.get("target_modules")

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=cfg.get("lora_dropout", 0),
    )

    if min_rank is None:
        min_rank = meta.get("min_rank", 1)

    add_tlora_adapter(
        model,
        lora_config,
        adapter_name=adapter_name,
        min_rank=min_rank,
        max_timestep=meta.get("max_timestep", 1000),
        ortho=meta.get("ortho", False),
        sig_type=meta.get("sig_type", "last"),
        ortho_init=meta.get("ortho_init", "random"),
        skip_ortho_init=True,
    )

    loaded, n_sd, missing = _load_weights_by_canonical_name(
        model, state_dict, adapter_name
    )
    if loaded == 0:
        raise RuntimeError(
            f"No adapter weights loaded from {path} (adapter_name={adapter_name}). "
            f"State dict has {len(state_dict)} keys."
        )
    if loaded < n_sd:
        warnings.warn(
            f"Loaded {loaded}/{n_sd} adapter keys from {path} (adapter_name={adapter_name}) "
            "— some keys did not match the model.",
            stacklevel=2,
        )
    if missing:
        warnings.warn(
            f"{len(missing)} adapter params for {adapter_name!r} in the model were NOT "
            f"overwritten by {path} — they keep their random init and will hurt "
            f"reproducibility. First missing: {missing[:3]}",
            stacklevel=2,
        )

    frozen_path = os.path.join(path, "tlora_frozen.pt")
    if os.path.isfile(frozen_path):
        _load_ortho_frozen(model, frozen_path, adapter_name)
    elif meta.get("ortho", False):
        warnings.warn(
            f"tlora_frozen.pt not found in {path} but adapter is ortho. "
            "Frozen SVD copies will be from a fresh random init — results may be incorrect.",
            stacklevel=2,
        )

    _ = autocast_adapter_dtype  # kept for API compatibility, intentionally unused


def cast_tlora_adapter_dtype(model: nn.Module, dtype: torch.dtype):
    """Cast all T-LoRA adapter parameters to *dtype* in-place.

    Touches ``lora_A``, ``lora_B`` for both variants and additionally
    ``lora_lambda``, ``base_A``, ``base_B``, ``base_lambda`` for ortho.
    Base layers (the frozen original weights) are **not** touched.

    Useful at inference time when running a mixed-precision base model
    (e.g. Flux in bf16) but wanting the LoRA branch to compute in higher
    precision for numerical fidelity — this matches the old T-LoRA Flux
    inference, which did ``lora_layers.to(dtype=torch.float32)`` after
    loading the checkpoint.
    """
    for module in model.modules():
        if not isinstance(module, _TLORA_LAYERS):
            continue
        for name in module.lora_A:
            module.lora_A[name].to(dtype=dtype)
            module.lora_B[name].to(dtype=dtype)
        if isinstance(module, TLoRAOrthoLinear):
            for name in module.lora_lambda:
                module.lora_lambda[name].data = module.lora_lambda[name].data.to(dtype=dtype)
            for name in module.base_A:
                module.base_A[name].to(dtype=dtype)
                module.base_B[name].to(dtype=dtype)
                module.base_lambda[name].data = module.base_lambda[name].data.to(dtype=dtype)


def set_adapter(model: nn.Module, adapter_names):
    """Activate one or more adapters (str or list of str)."""
    model.set_adapter(adapter_names)


def enable_adapter(model: nn.Module, name: str):
    """Activate a single adapter by name, un-disabling all layers first."""
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module._disable_adapters = False
    model.set_adapter(name)


def disable_adapter(model: nn.Module):
    """Disable all adapters on every LoRA layer."""
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module._disable_adapters = True


def remove_adapter(model: nn.Module, adapter_name: str):
    """Remove an adapter and free its memory."""
    model.delete_adapter(adapter_name)
    for module in model.modules():
        if isinstance(module, _TLORA_LAYERS):
            module.min_rank.pop(adapter_name, None)


def list_adapters(model: nn.Module) -> dict:
    """Return adapter names with their rank / min_rank info.

    Note:
        Reads from the *first* T-LoRA layer found; all layers are
        assumed to share the same adapter set.
    """
    for module in model.modules():
        if isinstance(module, _TLORA_LAYERS):
            info = {}
            for name in module.lora_A.keys():
                entry = {
                    "rank": module.r[name],
                    "min_rank": module.min_rank.get(name, 1),
                    "scaling": module.scaling[name],
                    "ortho": isinstance(module, TLoRAOrthoLinear),
                }
                info[name] = entry
            return info
    return {}
