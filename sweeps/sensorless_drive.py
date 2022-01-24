from natpn.sweep import schedule_slurm

base_options = dict(
    dataset="sensorless-drive",
    max_epochs=250,
    seed=[42, 137, 233, 330, 428],
    latent_dim=[4, 8, 16, 32, 64],
    learning_rate=[0.005, 0.001, 0.0005],
    use_learning_rate_decay=[True, False],
)

schedule_slurm(
    choices={**base_options, "flow_type": "radial", "flow_layers": [8, 16]},
    use_gpu=False,
)

schedule_slurm(
    choices={**base_options, "flow_type": "maf", "flow_layers": [1, 2, 4]},
    use_gpu=False,
)
