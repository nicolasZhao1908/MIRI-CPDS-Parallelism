#!/bin/bash
#SBATCH --job-name=submit-mpi.sh
#SBATCH -D .
#SBATCH --output=submit-mpi.sh.o%j
#SBATCH --error=submit-mpi.sh.e%j
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4

HOST=$(echo $HOSTNAME | cut -f 1 -d'.')
if [[ ${HOST} == *"6"* ]] || [[ ${HOST} == *"7"* ]] || [[ ${HOST} == *"8"* ]]
then
    echo "Use sbatch to execute this script"
    exit 0
fi

USAGE="\n USAGE: submit-mpi.sh PROG [nprocs] \n
        PROG     -> parallel program name\n
	nprocs -> number of MPI processes\n
	   nprocs is optional (by default 4 MPI processes)\n"

PROGRAM=$1
make $PROGRAM

MPI_PROCESSES=4

if (test $# -lt 1 || test $# -gt 2)
then
	echo -e $USAGE
    exit 0
fi

if [[ $# == 2 ]]
then
	MPI_PROCESSES=$2
fi

mpirun.mpich -np ${MPI_PROCESSES}  ./$PROGRAM > Output-$PROGRAM-$MPI_PROCESSES.txt

