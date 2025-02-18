#!/bin/bash
#SBATCH -w atlas-1-37                 # Request the specific node "compute-0-1"
#SBATCH --gres=gpu:3
#SBATCH --mem=64g
#SBATCH -c 48
#SBATCH --output=job_outputs/job_%j.out             # Redirect standard output to a file (job ID included)
#SBATCH --error=job_outputs/job_%j.err              # Redirect standard error to a file (job ID included)


# Your script commands start here.  This is what will run on compute-0-1.
echo "Starting job on $(hostname)"      # Print the hostname (should be compute-0-1)
date                                   # Print the current date and time
pwd                                    # print working directory

# Example: Run a Python script (replace with your actual command)
bash scripts/eval.sh

echo "Job finished"
date