#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=yp2201@nyu.edu
#SBATCH --output=slurm_%j.out

module purge    
singularity exec --nv --bind $SCRATCH/comp_gen --overlay $SCRATCH/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python $SCRATCH/comp_gen/comp_gen.py
"