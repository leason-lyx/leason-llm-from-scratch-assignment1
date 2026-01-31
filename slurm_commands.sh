# 查看当前作业队列状态
squeue -o "%.8i %.12P %.20j %.15u %.2t %.10M %.4D %R"

# 提交一个新的作业脚本
sbatch run.slurm

# 直接运行
srun --gres=gpu:1 -c 4 --mem=64G -t 00:20:00 uv run cs336_basics/generate.py