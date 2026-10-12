"""Muon for DiT hidden matrices with AdamW for the remaining parameters."""

from collections import ChainMap
import re

import torch


_HIDDEN_MATRIX = re.compile(
    r"(?:^|\.)(?:blocks|div_blocks)\.\d+\."
    r"(?:attn\.(?:q|k|v|out|qkv|proj)|mlp_fc[12]|mlp\.fc[12])\.weight$"
)


class MuonAdamW:
    """The optimizer interface used by TrainState, with two serialized children.

    The training step checks all gradients before calling either child and
    advances the EMA only once afterwards. Parameter groups remain the actual
    child dictionaries, including after restore, so LR scheduling reaches both.
    """

    optimizer_type = "muon_adamw"

    def __init__(self, net, *, lr, betas=(.9, .999), eps=1e-8,
                 momentum=.95, ns_steps=5, muon_lr_multiplier=1.,
                 adamw_lr_multiplier=1., weight_decay=1e-4):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("muon_adamw requires torch.optim.Muon")
        groups = {"muon": [], "adamw": []}
        names = {"muon": [], "adamw": []}
        for name, parameter in net.named_parameters():
            if not parameter.requires_grad:
                continue
            group = "muon" if parameter.ndim == 2 and _HIDDEN_MATRIX.search(name) else "adamw"
            groups[group].append(parameter)
            names[group].append(name)
        if not all(groups.values()):
            raise ValueError("muon_adamw requires both DiT hidden matrices and auxiliary parameters")
        self.parameter_names = names
        self.muon = torch.optim.Muon(
            [{"params": groups["muon"], "param_names": names["muon"],
              "lr_multiplier": muon_lr_multiplier, "group_name": "muon"}],
            lr=lr * muon_lr_multiplier, momentum=momentum, nesterov=True,
            ns_steps=ns_steps, eps=1e-7, adjust_lr_fn="match_rms_adamw",
            weight_decay=weight_decay,
        )
        self.adamw = torch.optim.AdamW(
            [{"params": groups["adamw"], "param_names": names["adamw"],
              "lr_multiplier": adamw_lr_multiplier, "group_name": "adamw"}],
            lr=lr * adamw_lr_multiplier, betas=betas, eps=eps, weight_decay=weight_decay,
        )

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def state(self):
        return ChainMap(self.muon.state, self.adamw.state)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"optimizer_type": self.optimizer_type, "version": 1,
                "parameter_names": self.parameter_names,
                "muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, saved):
        if saved.get("optimizer_type") != self.optimizer_type or saved.get("version") != 1:
            raise ValueError("Cannot restore another optimizer as Muon/AdamW; use a fresh optimizer")
        if saved.get("parameter_names") != self.parameter_names:
            raise ValueError("Muon/AdamW checkpoint parameter grouping does not match the model")
        self.muon.load_state_dict(saved["muon"])
        self.adamw.load_state_dict(saved["adamw"])
