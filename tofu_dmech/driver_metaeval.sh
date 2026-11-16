#!/bin/bash
# Auto-resubmit the TOFU meta-eval until it produces its table (MPS on this node crashes under sustained
# load; per-cell caching lets each fresh submit resume, and the final all-cells-cached run does pure-CPU
# analysis with no GPU -> cannot crash). Usage: driver_metaeval.sh <scale> <nfacts> [tries]
set -uo pipefail
cd "$HOME/gpu_8b_run/tofu_dmech"
scale=$1; nfacts=$2; tries=${3:-8}
rm -rf "cache_${scale}" "tofu_metaeval_table1_${scale}.csv" "tofu_metaeval_cells_${scale}.csv" "tofu_metaeval_fig4_${scale}.png"
for t in $(seq 1 "$tries"); do
  jid=$(sbatch --parsable --partition=gpu --extra=vram:70 --time=01:00:00 \
        --export=ALL,TOFU_SCALE=${scale},TOFU_NFACTS=${nfacts} run_metaeval.sbatch)
  echo "[driver ${scale}] try ${t}: job ${jid}"
  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -q .; do sleep 12; done
  if [ -f "tofu_metaeval_table1_${scale}.csv" ]; then
    echo "[driver ${scale}] DONE after ${t} tries"; break
  fi
  echo "[driver ${scale}] try ${t} ended; cached cells: $(ls cache_${scale} 2>/dev/null | tr '\n' ' ')"
done
echo "===== driver ${scale} finished ====="
[ -f "tofu_metaeval_table1_${scale}.csv" ] && echo "TABLE PRESENT" || echo "TABLE MISSING (exhausted tries)"
