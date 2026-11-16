#!/bin/bash
set -uo pipefail
cd "$HOME/gpu_8b_run/tofu_dmech"
nfacts=${1:-120}; tries=${2:-8}
rm -rf cache_tax_1b taxonomy_table_1b.csv taxonomy_rank_by_layer_1b.png
for t in $(seq 1 "$tries"); do
  jid=$(sbatch --parsable --partition=gpu --extra=vram:70 --time=01:00:00 \
        --export=ALL,TOFU_SCALE=1b,TOFU_NFACTS=${nfacts} run_taxonomy.sbatch)
  echo "[driver tax] try ${t}: job ${jid}"
  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -q .; do sleep 12; done
  if [ -f taxonomy_table_1b.csv ]; then echo "[driver tax] DONE after ${t} tries"; break; fi
  echo "[driver tax] try ${t} ended; cached: $(ls cache_tax_1b 2>/dev/null | tr '\n' ' ')"
done
echo "===== driver tax finished ====="
[ -f taxonomy_table_1b.csv ] && echo "TABLE PRESENT" || echo "TABLE MISSING"
