"""Defaults for new training and explicit density-stage recipes."""

# Missing optimizer fields in historical configs still mean RAdam in state_utils.
MUON_DEFAULTS = dict(
    optimizer="muon_adamw", learning_rate=1e-4, weight_decay=1e-4,
    schedule_type="sqrt", warmup_fraction=0., decay_steps=50000,
    b1=.9, b2=.999, eps=1e-8,
    muon_momentum=.95, muon_ns_steps=5,
    muon_lr_multiplier=1., adamw_lr_multiplier=1.,
)


# New-stage budgets and sqrt decay onset, without warmup.
# Applied before recipes and explicit CLI overrides; frozen run configs retain
# their recorded schedules.
PRETRAIN_DEFAULTS = {
    "pretrain_v": dict(total_steps=400000, decay_steps=50000),
    "pretrain_flowmap": dict(total_steps=100000, learning_rate=5e-4, decay_steps=10000),
}


# A3's archived recipe is adopted for AAAAAA and Chignolin velocity pretraining.
AAAAAA_PRETRAIN_V_DEFAULTS = dict(
    total_steps=400000, learning_rate=5e-4, weight_decay=1e-4,
    optimizer="muon_adamw", schedule_type="cosine", warmup_fraction=.05, bs=1024,
)


# The SCALLOP paper's molecular setup (arXiv:2606.29110, appendix): AdamW at 1e-4 with
# weight decay 1e-4 and gradient clip 10, square-root annealing after a 2.5% warm-up,
# EMA 0.999, batch 1000, 4e6 updates, learned e^{-w} loss weights (Eq. 24), times in
# [0, 1 - 1e-5], and a DiT with 6 shared and 3 log-density blocks. The paper leaves the
# sqrt decay onset and the weight network's width unstated; these keep the repo's 50k
# steps and EDM2's 128 log-variance channels.
PAPER_EPS = 1e-5
PAPER_DEFAULTS = dict(
    optimizer="adamw", learning_rate=1e-4, weight_decay=1e-4, clip=10.,
    schedule_type="sqrt", warmup_fraction=.025, decay_steps=50000,
    b1=.9, b2=.999, eps=1e-8, bs=1000, total_steps=4_000_000,
)
PAPER_LAYOUT = dict(num_layers=6, num_shared_layers=6, num_div_layers=3)
PAPER_LOGVAR_CHANNELS = 128


def apply_paper(cfg, *, mode):
    """The paper's stated setup, for the velocity teacher and the distilled flow map alike.

    Call after selecting the loss family and before explicit CLI overrides, which
    still apply (e.g. --total_steps, --offdiag_time_order).
    """
    if mode not in ("pretrain_v", "dlogpdt"):
        raise ValueError("the paper recipe covers --mode pretrain_v (teacher) and dlogpdt (SCALLOP)")
    cfg.training.recipe = "paper"
    cfg.training.tmax = 1. - PAPER_EPS
    cfg.training.diag_tmax = 1. - PAPER_EPS
    cfg.training.ema_facs = [.999]
    cfg.training.density_ema_fac = .999
    cfg.network.dit_kwargs.update(PAPER_LAYOUT)
    cfg.network.use_weight = True
    cfg.network.logvar_channels = PAPER_LOGVAR_CHANNELS
    cfg.teacher.ema_fac = .999
    cfg.optimization.update(PAPER_DEFAULTS)
    return cfg


def apply_branched_muon(cfg, *, mode, device):
    """Select branched density training with the current optimizer/schedule defaults.

    Call after selecting the loss family and before explicit CLI overrides.
    Parent/teacher paths remain caller-supplied; density checkpoints with a
    different branch layout are rejected by the existing checkpoint loader.
    """
    if mode != "dlogpdt" or cfg.training.loss_type not in ("lsd", "psd", "esd"):
        raise ValueError("branched_muon requires --mode dlogpdt and loss_type lsd, psd or esd")
    cfg.training.recipe = "branched_muon"
    cfg.training.cuda_graphs = str(device).split(":")[0] == "cuda"
    cfg.training.offdiag_time_order = "forward"
    cfg.training.diag_teacher_source = "external"
    cfg.training.offdiag_teacher_source = "ema" if cfg.training.loss_type == "psd" else "external"
    cfg.training.ema_facs = [.999]
    cfg.training.density_ema_fac = .999
    cfg.training.diag_tmax = .999
    cfg.network.dit_kwargs.update(num_layers=6, num_shared_layers=4, num_div_layers=2)
    cfg.network.predict_divergence = True
    cfg.network.vectorized_divergence = True
    cfg.network.div_loss_weighting_type = "linear"
    cfg.network.use_weight = False
    cfg.network.velocity_adaptive_power = 0.
    cfg.network.velocity_adaptive_eps = .01
    cfg.network.use_bfloat16 = False
    cfg.network.mixed_precision = "off"
    cfg.network.reset_optimizer = True
    cfg.network.reset_best_metrics = True
    cfg.teacher.ema_fac = .999
    cfg.optimization.update(MUON_DEFAULTS)
    cfg.optimization.update(clip=10., bs=1024, diag_fraction=.5, total_steps=400000)
    cfg.logging.update(log_freq=100, save_freq=10000, visual_freq=10000,
                       plot_bs=1000, n_sampling_steps=[4], importance_weight_clip=.002)
    return cfg
