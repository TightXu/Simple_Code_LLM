#!/usr/bin/env bash
# Overnight high-quality chain (authorized: user on 09-14 02:0x said "just do whatever you think is best from here, I am going to sleep")
#
# Why this order: tonight showed training-seed noise of +/-6 points, while the numbers we have been comparing are 78/79/80/81 -- a single-seed board
# is not enough to decide. So fix the "measurement" first, then talk about "gains":
#   stage 0 wait for the seed-soup chain (42/43/44 three-way average) to finish and be evaluated -> see whether the soup beats the mean of single seeds
#   stage 1 evaluation noise quantification: re-run 2 key models over 20 evaluation seeds (isolated run, does not pollute the main summary)
#          -> bootstrap the uncertainty of the "5-seed total" from it, giving every number on the main board an error bar
#   stage 2 5:1 recipe (data_mix_b147) add seeds 43/44 -> together with the existing seed 42, forms a 3-seed distribution
#   stage 3 1.5:1 recipe (data_mix_a44) add seeds 43/44 -> likewise a 3-seed distribution
#   stage 4 summary: report a "distribution" per recipe instead of a single point (3:1 already has 42/43/44)
# Serial use of the GPU throughout; every artifact lands in the logs; the main summary is only wrapped in "backup - run - restore", so no failing step should pollute it.
set -u
cd "$(dirname "$0")/.." || exit 1
exec > >(tee -a logs/chain_night_quality.log) 2>&1
say(){ echo "[$(date '+%F %T')] $*"; }
SEEDS20="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"

gpu_busy(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -i python || true; }
wait_no_gpu(){ local max=${1:-120} i=0
  while [ $i -lt $max ]; do [ -z "$(gpu_busy)" ] && return 0; sleep 30; i=$((i+1)); done
  say "  [warn] timed out waiting for the GPU, still busy: $(gpu_busy)"; return 1; }

train(){ [ -d "$2/final" ] && { say "already have $2, skipping"; return; }
  say "  training $2 (seed $3; data $1)"
  py -3.12 sft_train.py --bin-data "$1/sft_train.bin" --sft-val-bin "$1/sft_val.bin" \
    --val-bin ../data/all/val.bin --resume ../checkpoints_wsm/merged --ckpt-dir "$2" \
    --epochs 1 --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
    --batch-size 8 --grad-accum 4 --eval-every 200 --save-every 200 --val-batches 250 \
    --compile-mode default --seed "$3" > "logs/$2_train.log" 2>&1
  say "  $2 done: $(grep -m1 'Best val' "logs/$2_train.log" || echo see log)"; }

eval_arm(){ say "  [eval-sig] $2"
  cp -f eval/eval_sft_summary.json eval/_prev_sig.json 2>/dev/null || true
  py -3.12 eval_sft.py --models "$2" --skip-humaneval > "logs/eval_sig_$1.log" 2>&1
  say "     sig rc=$?"
  py -3.12 merge_eval.py --prev eval/_prev_sig.json --new eval/eval_sft_summary.json --out eval/eval_sft_summary.json
  cp -f eval/eval_sft_generations.jsonl "eval/generations_${1}_sig.jsonl" 2>/dev/null || true
  say "  [eval-nl]  $2"
  cp -f eval/nl_eval_summary.json eval/_prev_nl.json 2>/dev/null || true
  py -3.12 eval_nl.py --models "$2" > "logs/eval_nl_$1.log" 2>&1
  say "     nl rc=$?"
  py -3.12 merge_eval.py --prev eval/_prev_nl.json --new eval/nl_eval_summary.json --out eval/nl_eval_summary.json
  cp -f eval/nl_eval_generations.jsonl "eval/generations_${1}_nl.jsonl" 2>/dev/null || true; }

say "########## overnight chain start; C: free $(df -h /c | tail -1 | awk '{print $4}') ##########"

say "===== stage 0: wait for the seed-soup chain to finish evaluating SOUP-G8x3 ====="
i=0
while [ $i -lt 180 ]; do
  if [ -f eval/eval_sft_summary.json ] && grep -q "SOUP-G8x3-bestval" eval/eval_sft_summary.json; then break; fi
  sleep 60; i=$((i+1))
done
if grep -q "SOUP-G8x3-bestval" eval/eval_sft_summary.json; then say "  ready (waited $i minutes)"; else say "  [warn] timed out without SOUP-G8x3, continuing"; fi

say "===== stage 1: evaluation noise quantification (20 evaluation seeds, isolated run, no pollution of the main summary) ====="
noise_eval(){ local name="$1" tag="$2"
  say "  $name (20 seeds)"
  cp -f eval/eval_sft_summary.json eval/_keep_sig.json; cp -f eval/nl_eval_summary.json eval/_keep_nl.json
  py -3.12 eval_sft.py --models "$name" --seeds "$SEEDS20" --skip-humaneval > "logs/noise_sig_$tag.log" 2>&1
  say "     sig rc=$?"
  cp -f eval/eval_sft_summary.json "eval/noise20_${tag}_sig.json"
  cp -f eval/eval_sft_generations.jsonl "eval/generations_noise20_${tag}_sig.jsonl" 2>/dev/null || true
  cp -f eval/_keep_sig.json eval/eval_sft_summary.json
  py -3.12 eval_nl.py --models "$name" --seeds "$SEEDS20" > "logs/noise_nl_$tag.log" 2>&1
  say "     nl rc=$?"
  cp -f eval/nl_eval_summary.json "eval/noise20_${tag}_nl.json"
  cp -f eval/nl_eval_generations.jsonl "eval/generations_noise20_${tag}_nl.jsonl" 2>/dev/null || true
  cp -f eval/_keep_nl.json eval/nl_eval_summary.json
  local ns nn
  ns=$(py -3.12 -c "import json;print(len(json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))))")
  nn=$(py -3.12 -c "import json;print(len(json.load(open('eval/nl_eval_summary.json',encoding='utf-8'))['models']))")
  say "     main summary restored: sig $ns entries / NL $nn entries (should be 49+/49+; distinctly fewer means the restore failed)"; }
wait_no_gpu 60
noise_eval "G8-g3to1-bestval"  "g8bv"
noise_eval "SOUP-G8x3-bestval" "soup3"

say "===== stage 2: 5:1 recipe (data_mix_b147) add seeds 43/44 ====="
wait_no_gpu 60
train data_mix_b147 ckpt_g9_s43 43
eval_arm g9s43 "G9-g5to1-s43-final,G9-g5to1-s43-bestval"
train data_mix_b147 ckpt_g9_s44 44
eval_arm g9s44 "G9-g5to1-s44-final,G9-g5to1-s44-bestval"

say "===== stage 3: 1.5:1 recipe (data_mix_a44) add seeds 43/44 ====="
wait_no_gpu 60
train data_mix_a44 ckpt_g5_s43 43
eval_arm g5s43 "G5-g15-s43-final,G5-g15-s43-bestval"
train data_mix_a44 ckpt_g5_s44 44
eval_arm g5s44 "G5-g15-s44-final,G5-g15-s44-bestval"

say "===== stage 4: summary (recipe distributions + 20-seed noise) ====="
py -3.12 - <<'PY'
import json, itertools, random, statistics as st
def tot(m): return sum(int(v.split('/')[0]) for v in m['tasks'].values())
sig={m['model']:tot(m) for m in json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))}
nl={}
for m in json.load(open('eval/nl_eval_summary.json',encoding='utf-8'))['models']:
    nl.setdefault(m['model'],tot(m))

print("\n=== 3:1 recipe (data_mix_a88): distribution over three seeds ===")
for tag in ("", "-s43", "-s44"):
    n=f"G8-g3to1{tag}-bestval" if tag else "G8-g3to1-bestval"
    if n in sig: print(f"  {n:26s} NL {nl.get(n,'?')} / sig {sig[n]} = {(nl.get(n) or 0)+sig[n]}")
print("\n=== 5:1 recipe (data_mix_b147): distribution over three seeds ===")
for tag in ("", "-s43", "-s44"):
    n=f"G9-g5to1{tag}-bestval" if tag else "G9-g5to1-bestval"
    if n in sig: print(f"  {n:26s} NL {nl.get(n,'?')} / sig {sig[n]} = {(nl.get(n) or 0)+sig[n]}")
print("\n=== 1.5:1 recipe (data_mix_a44): distribution over three seeds ===")
for tag in ("", "-s43", "-s44"):
    n=f"G5-g15{tag}-final" if tag else "G5-g15-final"
    if n in sig: print(f"  {n:26s} NL {nl.get(n,'?')} / sig {sig[n]} = {(nl.get(n) or 0)+sig[n]}")

print("\n=== 20-seed noise quantification (same protocol as the 5-seed main board) ===")
random.seed(0)
for tag,label in (("g8bv","G8-g3to1-bestval(champion,seed42)"), ("soup3","SOUP-G8x3-bestval(three-way average)")):
    for col, fn, key in (("sig", f"eval/noise20_{tag}_sig.json", None), ("NL", f"eval/noise20_{tag}_nl.json", "models")):
        try:
            j=json.load(open(fn,encoding="utf-8"))
        except Exception as e:
            print(f"  {label} {col}: cannot read {fn} ({e})"); continue
        ms = j if isinstance(j,list) else j[key]
        m = ms[0]
        tasks=m["tasks"]
        totals={}
        for t,v in tasks.items():
            a,b=v.split("/"); totals[t]=int(a)
        ntask=len(totals)
        # use each task's success count over the 20 seeds to bootstrap the "5-seed" total-score distribution
        # per-task success rate estimate p=v/20; then resample for "5 seeds": 5 Bernoulli(p) draws per task
        boot=[]
        for _ in range(2000):
            s=0
            for v in totals.values():
                p=v/20.0
                s+=sum(1 for _ in range(5) if random.random()<p)
            boot.append(s)
        full=sum(totals.values())
        print(f"  {label} {col}: 20-seed total {full}/220 | 5-seed bootstrap median {st.median(boot):.0f} "
              f"| 5%~95% range {sorted(boot)[100]}~{sorted(boot)[1900]}")
PY
say "########## overnight chain done ##########"
