"""
T-LoRA layers: PEFT LoRA layers with timestep-dependent rank masking.

At diffusion timestep *t* the effective rank is::

    r_eff = int((max_t - t) / max_t * (rank - min_rank)) + min_rank

Only the first ``r_eff`` columns of lora_A's output are kept; the rest
are zeroed.  Higher timesteps (more noise) use fewer rank components,
which reduces overfitting on noisy inputs while allowing the model full
expressiveness at low-noise timesteps.

Two layer variants are provided:

* :class:`TLoRALinear` — standard LoRA (A/B) with timestep masking.
* :class:`TLoRAOrthoLinear` — orthogonal LoRA (Q/P/λ via SVD) with
  timestep masking.
"""

import copy
import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn

import inspect

from peft.tuners.lora.layer import LoraLayer

_UL_PARAMS = inspect.signature(LoraLayer.update_layer).parameters
_NEW_PEFT_API = "config" in _UL_PARAMS
_UPDATE_LAYER_HAS_BIAS = "lora_bias" in _UL_PARAMS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_timestep_mask(
    timestep: int,
    max_timestep: int,
    rank: int,
    min_rank: int,
    device: torch.device,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Compute a binary ``(1, rank)`` mask that zeroes out components above *r_eff*."""
    r_eff = int(((max_timestep - timestep) / max_timestep) * (rank - min_rank)) + min_rank
    r_eff = max(min_rank, min(rank, r_eff))
    mask = torch.zeros((1, rank), device=device, dtype=dtype)
    mask[:, :r_eff] = 1.0
    return mask


def _require_timestep(timestep: Optional[int]) -> int:
    """Raise if *timestep* has not been set yet."""
    if timestep is None:
        raise RuntimeError(
            "current_timestep is not set — call set_timestep() before forward"
        )
    return timestep


# ═══════════════════════════════════════════════════════════════════════════
# TLoRALinear
# ═══════════════════════════════════════════════════════════════════════════


class TLoRALinear(nn.Module, LoraLayer):
    """Drop-in replacement for PEFT ``lora.Linear`` with timestep-dependent
    rank masking.

    Register via ``LoraConfig._register_custom_module({nn.Linear: TLoRALinear})``
    before calling ``model.add_adapter(config)`` or ``get_peft_model(model, config)``.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self.max_timestep: int = 1000
        self.min_rank: dict[str, int] = {}
        self.current_timestep: Optional[int] = None

        self._active_adapter = adapter_name
        if _NEW_PEFT_API and config is not None:
            r = config.r
            self.update_layer(adapter_name, r, config.lora_alpha, config)
        else:
            _ul_kwargs = dict(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
            )
            if _UPDATE_LAYER_HAS_BIAS:
                _ul_kwargs["lora_bias"] = lora_bias
            self.update_layer(adapter_name, r, **_ul_kwargs)

    def _get_timestep_mask(
        self, rank: int, min_rank: int, device: torch.device, dtype: torch.dtype = None,
    ) -> torch.Tensor:
        t = _require_timestep(self.current_timestep)
        return _compute_timestep_mask(t, self.max_timestep, rank, min_rank, device, dtype)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_cast = x.to(lora_A.weight.dtype)

                after_A = lora_A(dropout(x_cast))
                rank = self.r[active_adapter]
                min_r = self.min_rank.get(active_adapter, 1)
                mask = self._get_timestep_mask(rank, min_r, after_A.device, after_A.dtype)
                after_A = after_A * mask

                result = result + lora_B(after_A) * scaling

            result = result.to(torch_result_dtype)

        return result

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args: Any,
        adapter_names: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = list(dict.fromkeys(adapter_names))
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append(
                [i for i, item in enumerate(adapter_names) if item == adapter]
            )

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            after_A = lora_A(dropout(sub_batch))
            rank = self.r[active_adapter]
            min_r = self.min_rank.get(active_adapter, 1)
            mask = self._get_timestep_mask(rank, min_r, after_A.device, after_A.dtype)
            after_A = after_A * mask

            lora_output = lora_B(after_A) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "tlora." + rep


# ═══════════════════════════════════════════════════════════════════════════
# TLoRAOrthoLinear
# ═══════════════════════════════════════════════════════════════════════════


class TLoRAOrthoLinear(nn.Module, LoraLayer):
    """Orthogonal T-LoRA layer: SVD-initialized Q/P/λ with timestep masking.

    The adapter contribution is::

        P(Q(dropout(x)) · λ · mask) − frozen_P(frozen_Q(dropout(x)) · frozen_λ · mask)

    At initialization the trainable and frozen copies are identical, so the
    net contribution starts at zero (like standard LoRA with B=0).  As Q, P,
    and λ are optimised the adapter drifts away from the frozen baseline.

    ``sig_type`` controls which singular components are kept after SVD:

    * ``"last"``  (default) — smallest singular values
    * ``"principal"`` — largest singular values
    * ``"middle"`` — middle singular values

    ``ortho_init`` controls the SVD source:

    * ``"random"`` (default) — SVD of a random Gaussian matrix
    * ``"base_layer"`` — SVD of the base layer's weight matrix

    Register via
    ``LoraConfig._register_custom_module({nn.Linear: TLoRAOrthoLinear})``
    before calling ``model.add_adapter(config)`` or
    ``get_peft_model(model, config)``.
    """

    # Defaults read during __init__.  ``get_tlora_peft_model`` temporarily
    # overrides these class attributes so that PEFT-created instances pick
    # up the user-specified values during SVD initialisation.
    _default_sig_type: str = "last"
    _default_ortho_init: str = "random"
    # When True, ``_ortho_init_adapter`` skips the costly SVD and fills
    # lora_A/B/lambda + frozen copies with zeros.  Used by
    # ``load_tlora_weights`` / ``load_adapter`` for inference — the loaded
    # checkpoint will overwrite everything anyway, so SVD is wasted work.
    _skip_init: bool = False

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self.max_timestep: int = 1000
        self.min_rank: dict[str, int] = {}
        self.current_timestep: Optional[int] = None
        self.sig_type: str = type(self)._default_sig_type
        self.ortho_init: str = type(self)._default_ortho_init

        self.lora_lambda = nn.ParameterDict({})
        self.base_A = nn.ModuleDict({})
        self.base_B = nn.ModuleDict({})
        self.base_lambda = nn.ParameterDict({})

        self._active_adapter = adapter_name
        if _NEW_PEFT_API and config is not None:
            r = config.r
            self.update_layer(adapter_name, r, config.lora_alpha, config)
        else:
            _ul_kwargs = dict(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
            )
            if _UPDATE_LAYER_HAS_BIAS:
                _ul_kwargs["lora_bias"] = lora_bias
            self.update_layer(adapter_name, r, **_ul_kwargs)
        self._ortho_init_adapter(adapter_name)

    # ------------------------------------------------------------------
    # SVD initialisation
    # ------------------------------------------------------------------

    def _ortho_init_adapter(self, adapter_name: str) -> None:
        """Re-initialise lora_A/lora_B from SVD and create λ + frozen copies.

        When ``type(self)._skip_init`` is set (during adapter loading) the
        expensive SVD is skipped and lora_A/B/λ are initialised randomly.
        The caller is then expected to overwrite every parameter (both
        trainable and frozen copies) from a saved state dict.
        """
        r = self.r[adapter_name]
        A = self.lora_A[adapter_name]
        B = self.lora_B[adapter_name]
        in_features = A.weight.shape[1]
        out_features = B.weight.shape[0]

        if type(self)._skip_init:
            # Leave ``A.weight`` / ``B.weight`` as PEFT's ``update_layer``
            # set them (kaiming / zeros) and use a tiny random λ —
            # everything will be overwritten when the checkpoint loads.
            lam = torch.randn(
                (1, r), dtype=A.weight.dtype, device=A.weight.device
            ) * 0.01
        else:
            # Run SVD on CUDA if available — it is orders of magnitude
            # faster than CPU.  The resulting tensors are cast back to the
            # target module's device/dtype after selection.
            svd_device = A.weight.device
            if svd_device.type == "cpu" and torch.cuda.is_available():
                svd_device = torch.device("cuda")

            if self.ortho_init == "base_layer":
                weight = self.get_base_layer().weight.data.float().to(svd_device)
                u, s, vh = torch.linalg.svd(weight, full_matrices=True)
                q_w, p_w, lam = self._pick_components(vh, u, s, r)
                del u, s, vh, weight
            else:
                base_m = torch.normal(
                    mean=0, std=1.0 / r,
                    size=(in_features, out_features),
                    device=svd_device,
                )
                u, s, vh = torch.linalg.svd(base_m, full_matrices=True)
                q_w, p_w, lam = self._pick_components(u, vh, s, r)
                del u, s, vh, base_m

            A.weight.data = q_w.to(dtype=A.weight.dtype, device=A.weight.device)
            B.weight.data = p_w.to(dtype=B.weight.dtype, device=B.weight.device)
            lam = lam.to(dtype=A.weight.dtype, device=A.weight.device)

        self.lora_lambda[adapter_name] = nn.Parameter(lam)

        self.base_A[adapter_name] = copy.deepcopy(A)
        self.base_B[adapter_name] = copy.deepcopy(B)
        self.base_lambda[adapter_name] = nn.Parameter(lam.clone(), requires_grad=False)

        self.base_A[adapter_name].requires_grad_(False)
        self.base_B[adapter_name].requires_grad_(False)

        for param in self.parameters():
            param.data = param.data.contiguous()

    def _pick_components(self, q_src, p_src, s, r):
        """Select *r* singular components from SVD results.

        For random init:  q_src=U  (in×in),  p_src=Vh (out×out)
        For base_layer:   q_src=Vh (in×in),  p_src=U  (out×out)

        Returns ``(q_weight, p_weight, lambda_)`` with shapes
        ``(r, in)``, ``(out, r)``, ``(1, r)``.
        """
        n_components = min(s.shape[0], q_src.shape[0], p_src.shape[1])
        r = min(r, n_components)

        if self.sig_type == "principal":
            q_w = q_src[:r].clone()
            p_w = p_src[:, :r].clone()
            lam = s[None, :r].clone()
        elif self.sig_type == "last":
            q_w = q_src[-r:].clone()
            p_w = p_src[:, -r:].clone()
            lam = s[None, -r:].clone()
        else:  # middle
            start = math.ceil((q_src.shape[0] - r) / 2)
            q_w = q_src[start : start + r].clone()
            start = math.ceil((p_src.shape[1] - r) / 2)
            p_w = p_src[:, start : start + r].clone()
            start = math.ceil((s.shape[0] - r) / 2)
            lam = s[None, start : start + r].clone()
        return q_w, p_w, lam

    # ------------------------------------------------------------------
    # Timestep mask
    # ------------------------------------------------------------------

    def _get_timestep_mask(
        self, rank: int, min_rank: int, device: torch.device, dtype: torch.dtype = None,
    ) -> torch.Tensor:
        t = _require_timestep(self.current_timestep)
        return _compute_timestep_mask(t, self.max_timestep, rank, min_rank, device, dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _ortho_adapter_forward(self, x, active_adapter, mask):
        """``P(Q(dropout(x))·λ·mask) − base_P(base_Q(dropout(x))·base_λ·mask)``"""
        dropout = self.lora_dropout[active_adapter]
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        lam = self.lora_lambda[active_adapter]
        scaling = self.scaling[active_adapter]

        x_drop = dropout(x)
        after_A = lora_A(x_drop) * lam * mask
        adapter_out = lora_B(after_A)

        base_after_A = self.base_A[active_adapter](x_drop) * self.base_lambda[active_adapter] * mask
        base_out = self.base_B[active_adapter](base_after_A)

        return (adapter_out - base_out) * scaling

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                x_cast = x.to(self.lora_A[active_adapter].weight.dtype)
                rank = self.r[active_adapter]
                min_r = self.min_rank.get(active_adapter, 1)
                mask = self._get_timestep_mask(rank, min_r, x_cast.device, x_cast.dtype)

                result = result + self._ortho_adapter_forward(x_cast, active_adapter, mask)

            result = result.to(torch_result_dtype)

        return result

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args: Any,
        adapter_names: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = list(dict.fromkeys(adapter_names))
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append(
                [i for i, item in enumerate(adapter_names) if item == adapter]
            )

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            sub_batch = x[sub_batch_indices_list[i]].to(
                self.lora_A[active_adapter].weight.dtype
            )
            rank = self.r[active_adapter]
            min_r = self.min_rank.get(active_adapter, 1)
            mask = self._get_timestep_mask(rank, min_r, sub_batch.device, sub_batch.dtype)

            lora_output = self._ortho_adapter_forward(sub_batch, active_adapter, mask)
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "tlora_ortho." + rep
