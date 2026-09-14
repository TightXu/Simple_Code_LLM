#!/usr/bin/env bash
# Round 3: extend four single-seed arms to the 20-seed precision protocol (same seed set as the existing 6 measurements -> mutually comparable)
# Goal: obtain the precision level of every seed for the 1.5:1 / 3:1 / 5:1 recipes, so as to answer "which recipe is truly stronger",
#       instead of being misled by a single seed's lucky draw.
set -u
cd "/c/Users/Tight/Documents/Train LLM/435M-v2/fine-tuning" || exit 1
exec > >(tee -a logs/chain_seed_20seeds.log) 2>&1
say(){ echo "[$(date '+%F %T')] $*"; }
SEEDS20="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"

score20(){ local name="$1" tag="$2"
  [ -f "eval/noise20_${tag}_sig.json" ] && [ -f "eval/noise20_${tag}_nl.json" ] && { say "  already have $tag, skipping"; return; }
  say "  20-seed: $name"
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
  say "     main summary restored: $(py -3.12 -c "import json;print(len(json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))))") entries"; }

say "########## round 3: 20-seed precision measurement of single-seed arms ##########"
score20 "G8-g3to1-s43-bestval" g8s43
score20 "G8-g3to1-s44-bestval" g8s44
score20 "G9-g5to1-s43-bestval" g9s43
score20 "G9-g5to1-s44-bestval" g9s44

say "== summary: seed distribution per recipe (20-seed precision protocol) =="
py -3.12 - <<'PY'
import json, glob, os
groups = {
 "1.5:1 (a44)":  [("G5-g15-bestval(seed42)","g5bv"), ("G5-g15-final(seed42, chose final)","g5x3fi")],
 "3:1 (a88)":    [("seed42","g8bv"), ("seed43","g8s43"), ("seed44","g8s44"), ("three-way soup","soup3")],
 "5:1 (b147)":   [("seed42","g9bv"), ("seed43","g9s43"), ("seed44","g9s44"), ("three-way soup","g9x3bv")],
}
def load(tag):
    try:
        s=json.load(open(f"eval/noise20_{tag}_sig.json",encoding="utf-8")); s=s if isinstance(s,list) else s["models"]
        n=json.load(open(f"eval/noise20_{tag}_nl.json",encoding="utf-8")); n=n if isinstance(n,list) else n["models"]
    except Exception:
        return None
    def v(ms):
        t=ms[0]["tasks"]; return sum(int(x.split("/")[0]) for x in t.values()), sum(int(x.split("/")[1]) for x in t.values())
    sa,sb=v(s); na,nb=v(n)
    return na/nb*55, sa/sb*55
for rec, items in groups.items():
    print(f"\n=== {rec} ===")
    vals=[]
    for label, tag in items:
        r=load(tag)
        if r is None: print(f"  {label:34s} (not measured)"); continue
        vals.append(r[0]+r[1]); print(f"  {label:34s} NL {r[0]:5.1f} / sig {r[1]:5.1f} = {r[0]+r[1]:5.1f}")
    if vals:
        import statistics as st
        print(f"  -> mean {st.mean(vals):.1f}  range {max(vals)-min(vals):.1f}  n={len(vals)}")
PY
say "########## round 3 done ##########"
