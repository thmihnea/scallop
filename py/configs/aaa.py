"""Current aaa configuration; internal data/checkpoint identities are preserved."""
import os

from ml_collections import ConfigDict

experiments = [
    ("lsd", None, "stopped"),
    ("psd", "uniform", "stopped"),
    ("psd", "midpoint", "stopped"),
    ("esd", None, "stopped"),
]


def get_config(slurm_id: int, dataset_location: str = "", output_folder: str = ""):
    loss, psd, stopgrad = experiments[slurm_id % len(experiments)]
    method = f"{loss}_{psd}" if psd else loss
    name = f"ala3_implicit_dit_{method}"
    return ConfigDict(
        dict(
            training=dict(
                stopgrad_type=stopgrad,
                psd_type=psd,
                loss_type=loss,
                tmin=0.,
                tmax=1.,
                seed=42,
                cuda_graphs=None,
                ema_facs=[.999],
                ndevices=1,
                diag_teacher_source="external",
                offdiag_teacher_source="external",
                offdiag_time_order="forward",
                density_ema_fac=.999,
                diag_tmax=.999,
            ),
            problem=dict(
                d=99,
                n_particles=33,
                n_dimension=3,
                target="ala3_implicit",
                ts_data_dir=dataset_location,
                temperature=310.0,
                interp_type="linear",
                base="mean-free-gaussian",
                com_augmentation=True,
            ),
            optimization=dict(
                bs=1024,
                diag_fraction=.5,
                clip=10.,
                total_steps=100000,
                optimizer="muon_adamw",
                learning_rate=1e-4,
                weight_decay=1e-4,
                schedule_type="sqrt",
                warmup_fraction=0.,
                decay_steps=50000,
                b1=.9,
                b2=.999,
                eps=1e-8,
                muon_momentum=.95,
                muon_ns_steps=5,
                muon_lr_multiplier=1.,
                adamw_lr_multiplier=1.,
            ),
            logging=dict(
                plot_bs=1000,
                log_freq=100,
                visual_freq=10000,
                save_freq=10000,
                wandb_project="aaa-scallop",
                wandb_name=name,
                wandb_entity=os.getenv("WANDB_ENTITY", "RKOY"),
                output_folder=output_folder,
                output_name=name,
                importance_weight_clip=.002,
                n_sampling_steps=[4],
                visual_ema_factor=None,
                res_dir=output_folder,
            ),
            network=dict(
                com_aware=True,
                network_type="dit",
                load_path="",
                n_particles=33,
                n_dimension=3,
                input_dims=(33, 3),
                rescale=1.,
                divergence_rescale=0.,
                predict_divergence=True,
                vectorized_divergence=True,
                divergence_mode="hutchinson",
                div_loss_weighting_type="uniform",
                sigma_data=1.,
                load_ema_fac=None,
                logvar_channels=None,
                reset_optimizer=True,
                use_bfloat16=False,
                mixed_precision="off",
                use_weight=False,
                dit_kwargs=dict(
                    d_model=192,
                    num_layers=6,
                    num_shared_layers=4,
                    num_div_layers=2,
                    nhead=6,
                    mlp_ratio=4.,
                    time_embedding_dim=64,
                ),
            ),
            teacher=dict(
                load_path="",
                ema_fac=.999,
            ),
        )
    )
