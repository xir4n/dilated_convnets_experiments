import os

script_name = "main"
script_path = f'{os.getenv("SCRATCH")}/dilated_convnets_experiments/main.py'
project_name = "optimizers"
save_foler = f'{os.getenv("SCRATCH")}/outputs/dilated_convnets_experiments/{project_name}'
arch = "MuReNN"

# Model hyperparameters
Q = 4
T = 4
J = 6
lrs = [1e-1, 1e-2, 1e-3, 1e-4]
scale_factors = [0.707, 1, 1.414, 2]
optimizers = ["SGD", "Adam"]
# Dataset hyperparameters
num_samples = 1000
batch_size = 256
seg_length = 2**10
step_mins = [1, 8]
step_maxs = [2, 16]


# Create folder.
sbatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), project_name)
os.makedirs(sbatch_dir, exist_ok=True)

experiment_names = []
for i, step_min in enumerate(step_mins):
    for scale_factor in scale_factors:
        for lr in lrs:
            for optimizer in optimizers:
                step_max = step_maxs[i]
                experiment_name = f"s{scale_factor}_f{step_min}-{step_max}_{optimizer}.replace('.', '_')"
                experiment_names.append(experiment_name)
                file_name = experiment_name + ".sbatch"
                file_path = os.path.join(sbatch_dir, file_name)
                sav_dir = os.path.join(save_foler, experiment_name)
                
                # Generate file.
                with open(file_path, "w") as f:
                    cmd_args = [
                        script_path, 
                        f"--save_foler {sav_dir}",
                        f"--arch {arch}",
                        f"--Q {Q}",
                        f"--T {T}",
                        f"--J {J}",
                        f"--lr {lr}",
                        f"--scale_factor {scale_factor}",
                        f"--num_samples {num_samples}",
                        f"--batch_size {batch_size}",
                        f"--seg_length {seg_length}",
                        f"--step_min {step_min}",
                        f"--step_max {step_max}",
                        f"--optimizer {optimizer}",
                    ]

                    f.write("#!/bin/bash\n")
                    f.write("\n")
                    f.write("#BATCH --job-name=" + experiment_name + "\n")
                    f.write("#SBATCH --nodes=1\n")
                    f.write("#SBATCH -C v100-32g\n")
                    f.write("#SBATCH --tasks-per-node=1\n")
                    f.write("#SBATCH --gres=gpu:1\n")
                    f.write("#SBATCH --cpus-per-task=10\n")
                    f.write("#SBATCH --hint=nomultithread\n")

                    f.write("#SBATCH --time=1:00:00\n")
                    f.write("#SBATCH --account=nvz@v100\n")
                    f.write("#SBATCH --output=" + experiment_name + "_%j.out\n")
                    f.write("\n")
                    f.write("module purge\n")
                    f.write("\n")
                    f.write("module load anaconda-py3/2023.09\n")
                    f.write("\n")
                    f.write("export PATH=$WORK/.local/bin:$PATH\n")
                    f.write("conda activate murenn_fb\n")
                    f.write(" ".join(["python"] + cmd_args) + "\n")
                    f.write("\n")


# Open shell file.
file_path = os.path.join(sbatch_dir, script_name.split("_")[0] + ".sh") 
with open(file_path, "w") as f:
    for experiment_name in experiment_names:
        file_name = experiment_name + ".sbatch"
        file_path = os.path.join(sbatch_dir, file_name)
        sbatch_str = "sbatch " + file_path
        f.write(sbatch_str + "\n")

mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)