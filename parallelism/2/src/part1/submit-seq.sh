#!/bin/bash

#SBATCH --job-name=submit-heat-seq.sh
#SBATCH -D .
#SBATCH --output=gauss_1024.out
#SBATCH --error=error%j.err
## #SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


PROGRAM=heat

make $PROGRAM

./$PROGRAM test.dat
