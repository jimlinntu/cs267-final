### To allocate a node (bridge)
```
salloc -N 1 -n 1 -p GPU-shared -t 00:30:00 --gres=gpu:1
```

### To allocate a node (cori)
```
module load cgpu; module load cuda/11.4.0
salloc -C gpu -t 1:00:00 -c 10 -G 1 -q interactive -A mp309
```