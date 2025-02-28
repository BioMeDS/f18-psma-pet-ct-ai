import tensorboard_reducer as tbr
import sys
import os

input_dir = sys.argv[1]
output_dir = sys.argv[2]

tb = tbr.load_tb_events([input_dir])

os.makedirs(output_dir, exist_ok=True)

for k in tb.keys():
  tb[k].to_csv(f'{output_dir}/{k}.csv')

# Run in loop via
#for i in model*; do python ~/projects/f18-psma-pet-ct-ai/code/tb2csv.py $i tsv_log/$i ; done
# Then get maximum for each run in tsv_log via
#for i in model*; do echo -n "$i\t"; cat $i/val_acc.csv | tail -n+2 | tr , "\t" | awk '$2 > max {max=$2; maxline=$0} END {print maxline}'; done >val_acc_maxima.tsv
