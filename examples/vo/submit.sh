#!/bin/bash

bsub << EOF
#BSUB -J calibrate-household-parallel
#BSUB -n 20
#BSUB -R "span[ptile=20] affinity[core(1)]"
#BSUB -q tuleta
#BSUB -oo output-parallel.txt
#BSUB -W 4:00
#BSUB -x

export PYTHONPATH=\$(pwd)
python examples/vo/parallel.py
EOF

#bsub << EOF
#BSUB -J calibrate-household-serial
#BSUB -n 1
#BSUB -q tuleta
#BSUB -oo output-serial.txt
#BSUB -x
#
#export PYTHONPATH=\$(pwd)
#python examples/vo/run.py
#EOF
