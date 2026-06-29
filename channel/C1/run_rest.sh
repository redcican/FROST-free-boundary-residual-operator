cd /home/cican/paper/new_idea/FROST/channel/C1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate base
export TORCH_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# wait for the running phi/random to finish first (only ever one training at a time)
while pgrep -f "[t]rain_c1_channel.py" >/dev/null; do sleep 20; done
for cfg in "gamma random" "phi topo" "gamma topo"; do
  set -- $cfg
  echo "=== $1 $2 ==="
  python train_c1_channel.py --rep $1 --split $2 --epochs 100
  sleep 3
done
echo ALLREST_DONE
