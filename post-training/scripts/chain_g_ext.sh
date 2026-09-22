#!/usr/bin/env bash
# Gold-standard ratios after pool expansion: 5:1 and 8:1 (protocol identical to G3/G5/G7/G8 --min-score 0.9, sources = shard0-13)
#   G9  = gold 147.0M : signature 29.4M ~ 5:1
#   G10 = gold 235.2M : signature 29.4M ~ 8:1 (capped by the pool size, extract as much as we can)
set -u
cd "$(dirname "$0")/.." || exit 1
exec > >(tee -a logs/chain_g_ext.log) 2>&1
say(){ echo "[$(date '+%F %T')] $*"; }
SRC="data_instr_raw/self_oss_instruct_50k.parquet"
for s in 0 01 02 03 04 05 06 07 08 09 10 11 12 13; do
  f="data_instr_raw/opencodeinstruct_shard${s}.parquet"
  [ -s "$f" ] && SRC="$SRC $f" || say "WARNING: missing shard $f (skipped)"
done
say "source file list: $(echo $SRC | tr ' ' '\n' | wc -l) files"

build_one(){ [ -f "$1/instr_train.bin" ] && { say "already have $1, skipping"; return; }
  say "building $1 (target $2 tokens)"
  py -3.12 build_instr_data.py --sources $SRC --out-dir "$1" --target-tokens "$2" \
     --crossing-policy keep --min-score 0.9 2>&1 | tail -2; }
mix(){ [ -f "$1/sft_train.bin" ] && { say "already have $1, skipping"; return; }
  say "mixing $1"; py -3.12 mix_bins.py --out "$1" --parts data/sft_train.bin "$2/instr_train.bin" \
     --val-parts data/sft_val.bin "$2/instr_val.bin" --mode interleave --block-chunks 256 2>&1 | tail -1; }
train(){ [ -d "$2/final" ] && { say "already have $2, skipping"; return; }
  say "training $2"
  py -3.12 sft_train.py --bin-data "$1/sft_train.bin" --sft-val-bin "$1/sft_val.bin" \
    --val-bin ../data/all/val.bin --resume ../checkpoints_wsm/merged --ckpt-dir "$2" \
    --epochs 1 --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
    --batch-size 8 --grad-accum 4 --eval-every 400 --save-every 400 --val-batches 250 \
    --compile-mode default --seed 42 > "logs/$2_train.log" 2>&1
  say "$2 done: $(grep -m1 'Best val' "logs/$2_train.log" || echo see log)"; }
eval_arm(){ say "  [eval-sig] $2"
  cp -f eval/eval_sft_summary.json eval/_prev_sig.json 2>/dev/null || true
  py -3.12 eval_sft.py --models "$2" --skip-humaneval > "logs/eval_sig_$1.log" 2>&1
  py -3.12 merge_eval.py --prev eval/_prev_sig.json --new eval/eval_sft_summary.json --out eval/eval_sft_summary.json
  cp -f eval/eval_sft_generations.jsonl "eval/generations_${1}_sig.jsonl" 2>/dev/null || true
  say "  [eval-nl]  $2"
  cp -f eval/nl_eval_summary.json eval/_prev_nl.json 2>/dev/null || true
  py -3.12 eval_nl.py --models "$2" > "logs/eval_nl_$1.log" 2>&1
  py -3.12 merge_eval.py --prev eval/_prev_nl.json --new eval/nl_eval_summary.json --out eval/nl_eval_summary.json
  cp -f eval/nl_eval_generations.jsonl "eval/generations_${1}_nl.jsonl" 2>/dev/null || true; }

say "=========== expanded-pool ratio sweep (5:1 -> 8:1) start ==========="
build_one data_instr_b147 147000000; mix data_mix_b147 data_instr_b147; train data_mix_b147 ckpt_g9
eval_arm g9 "G9-g5to1-final,G9-g5to1-bestval"
build_one data_instr_b235 235200000; mix data_mix_b235 data_instr_b235; train data_mix_b235 ckpt_g10
eval_arm g10 "G10-g8to1-final,G10-g8to1-bestval"
say "=========== expanded-pool sweep done ==========="
py -3.12 - <<'PY'
import json
def tot(m): return sum(int(v.split('/')[0]) for v in m['tasks'].values())
sig={m['model']:tot(m) for m in json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))}
nl={}
for m in json.load(open('eval/nl_eval_summary.json',encoding='utf-8'))['models']: nl.setdefault(m['model'],tot(m))
rows=[(n,nl.get(n,0),sig.get(n,0)) for n in set(sig)|set(nl)]
print('model'.ljust(24),'NL','sig','total')
for n,a,b in sorted(rows,key=lambda r:-(r[1]+r[2]))[:10]: print(f'{n:24s} {a:>3} {b:>4} {a+b:>5}')
PY
