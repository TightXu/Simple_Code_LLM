#!/usr/bin/env bash
# Gold-standard chain: G1 (pure gold 62.5M) -> G2 (gold + signature 91.9M), each step trains + evaluates both columns + merges
# Usage: once the GPU is free, run bash chain_g.sh   (completed steps are skipped, safe to re-run)
set -u
cd "/c/Users/Tight/Documents/Train LLM/435M-v2/fine-tuning" || exit 1
exec > >(tee -a logs/chain_g.log) 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

eval_arm() {
  local short="$1" models="$2"
  say "  [eval-sig] $models"
  cp -f eval/eval_sft_summary.json eval/_prev_sig.json 2>/dev/null || true
  py -3.12 eval_sft.py --models "$models" --skip-humaneval > "logs/eval_sig_${short}.log" 2>&1
  py -3.12 merge_eval.py --prev eval/_prev_sig.json --new eval/eval_sft_summary.json --out eval/eval_sft_summary.json
  cp -f eval/eval_sft_generations.jsonl "eval/generations_${short}_sig.jsonl" 2>/dev/null || true
  say "  [eval-nl]  $models"
  cp -f eval/nl_eval_summary.json eval/_prev_nl.json 2>/dev/null || true
  py -3.12 eval_nl.py --models "$models" > "logs/eval_nl_${short}.log" 2>&1
  py -3.12 merge_eval.py --prev eval/_prev_nl.json --new eval/nl_eval_summary.json --out eval/nl_eval_summary.json
  cp -f eval/nl_eval_generations.jsonl "eval/generations_${short}_nl.jsonl" 2>/dev/null || true
}

say "=========== gold-standard chain start (G1 -> G2) ==========="

if [ -d ckpt_g1/final ]; then say "G1 exists, skipping"; else
  say "G1 training (pure gold 62.5M, ~32 min)"
  bash run_sft_g1.sh > logs/g1_train.log 2>&1
  say "G1 done: $(grep -m1 'Best val' logs/g1_train.log || echo see log)"
fi
eval_arm g1 "G1-gold-final,G1-gold-bestval"

if [ -d ckpt_g2/final ]; then say "G2 exists, skipping"; else
  say "G2 training (gold + signature 91.9M, ~48 min)"
  bash run_sft_g2.sh > logs/g2_train.log 2>&1
  say "G2 done: $(grep -m1 'Best val' logs/g2_train.log || echo see log)"
fi
eval_arm g2 "G2-goldsig-final,G2-goldsig-bestval"

say "=========== gold-standard chain done ==========="
py -3.12 - <<'PY'
import json
def tot(m): return sum(int(v.split('/')[0]) for v in m['tasks'].values())
sig=json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))
nl=json.load(open('eval/nl_eval_summary.json',encoding='utf-8'))['models']
smap={m['model']:tot(m) for m in sig}
nmap={}
for m in nl: nmap.setdefault(m['model'], tot(m))
rows=[(n,nmap.get(n,0),smap.get(n,0)) for n in set(smap)|set(nmap)]
print('model'.ljust(24),'NL','sig','total')
for n,a,b in sorted(rows,key=lambda r:-(r[1]+r[2]))[:10]:
    print(f'{n:24s} {a:>3} {b:>4} {a+b:>5}')
PY
