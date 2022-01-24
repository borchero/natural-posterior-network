import os
import uuid
from itertools import product
from pathlib import Path
from typing import Any, Dict, Union

Primitive = Union[str, int, float, bool]


def schedule_slurm(
    choices: Dict[str, Any],
    use_gpu: bool = False,
) -> None:
    """
    Runs the training script for the provided choices.

    Args:
        choices: The choices to run the hyperparameter search for.
        use_gpu: Whether to use a GPU for training.
    """
    # Get all combinations from the provided choices
    combinations = product(
        *[
            [(key, o) for o in options] if isinstance(options, list) else [(key, options)]
            for key, options in choices.items()
        ]
    )
    configs = []
    for combination in combinations:
        config = {}
        for key, value in combination:
            config[key] = value
        configs.append(config)

    # Generate CLI flags from the combinations
    flags = [" ".join([f"--{key} {value}" for key, value in config.items()]) for config in configs]

    # Store CLI flags in cache file
    unique_id = uuid.uuid4()
    path = Path.home() / f".cache/slurm/{unique_id}.options"
    path.parent.mkdir(exist_ok=True)
    with path.open("w+", encoding="utf-8") as f:
        f.write("\n".join(flags))
    print(f"Saved options to {str(path)}.")

    # Schedule array job
    script = f"{Path(__file__).parent.parent}/sweeps/slurm/{'gpu' if use_gpu else 'cpu'}.sh"
    os.system(f"sbatch --array=1-{len(configs)} {script} {path.absolute()}")

    # Print success
    print(f"Successfully scheduled {len(configs)} jobs.")
