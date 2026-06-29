set -e
cd /home/cican/paper/new_idea/FROST/channel/C1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate base
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for rep in phi gamma; do
  for split in random topo; do
    echo "=== $rep $split ==="
    python train_c1_channel.py --rep $rep --split $split --epochs 100
  done
done
echo ALLDONE_CHANNEL
