#!/usr/bin/env bash
# Round 4: turn the "best recipe (5:1 = data_mix_b147)" into a robust model
#   evidence (20-seed precision protocol): 5:1 three seeds = 80.2 / 72.0 / 81.0 (mean 77.7, highest overall),
#                             3:1 = 76.0 / 72.0 / 70.8 (mean 72.9), 1.5:1 soup = 75.5
#   (1) top-two soup (s42 + s44, both 80+) -- note: this carries a suspicion of selection on the evaluation set, must be flagged in the report
#   (2) add 2 more seeds (45/46, no selection bias) -> five-way soup, purely variance reduction + a more trustworthy recipe mean
#   (3) evaluate everything with the same 20-seed set, comparable with existing measurements
set -u
cd "$(dirname "$0")/.." || exit 1
exec > >(tee -a logs/chain_g9_soup_final.log) 2>&1
say(){ echo "[$(date '+%F %T')] $*"; }
SEEDS20="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"

train(){ [ -d "$2/final" ] && { say "  already have $2, skipping"; return; }
  say "  training $2 (seed $3; data $1)"
  py -3.12 sft_train.py --bin-data "$1/sft_train.bin" --sft-val-bin "$1/sft_val.bin" \
    --val-bin ../data/all/val.bin --resume ../checkpoints_wsm/merged --ckpt-dir "$2" \
    --epochs 1 --lr 2e-5 --warmup-steps 50 --lr-schedule cosine --eta-min-factor 0.1 \
    --batch-size 8 --grad-accum 4 --eval-every 200 --save-every 200 --val-batches 250 \
    --compile-mode default --seed "$3" > "logs/$2_train.log" 2>&1
  say "  $2 done: $(grep -m1 'Best Val' "logs/$2_train.log" || grep -m1 'Best val' "logs/$2_train.log" || echo see log)"; }

eval_arm(){ say "  [eval-sig] $2"
  cp -f eval/eval_sft_summary.json eval/_prev_sig.json 2>/dev/null || true
  py -3.12 eval_sft.py --models "$2" --skip-humaneval > "logs/eval_sig_$1.log" 2>&1; say "     sig rc=$?"
  py -3.12 merge_eval.py --prev eval/_prev_sig.json --new eval/eval_sft_summary.json --out eval/eval_sft_summary.json
  say "  [eval-nl]  $2"
  cp -f eval/nl_eval_summary.json eval/_prev_nl.json 2>/dev/null || true
  py -3.12 eval_nl.py --models "$2" > "logs/eval_nl_$1.log" 2>&1; say "     nl rc=$?"
  py -3.12 merge_eval.py --prev eval/_prev_nl.json --new eval/nl_eval_summary.json --out eval/nl_eval_summary.json; }

build3(){  # $1=type $2=three arm prefixes $3=output name (chained: A+B 0.5 -> +C 1/3)
  local SUB="$1" OUT="$3"; set -- $2
  [ -f "ck_soup/$OUT/checkpoint.pt" ] && { say "  already have ck_soup/$OUT"; return; }
  say "  three-way average -> ck_soup/$OUT"
  py -3.12 soup_models.py --a "$1/$SUB" --b "$2/$SUB" --alphas 0.5 --label "tmp_${OUT}a_" --out-dir ck_soup > "logs/soup_${OUT}_1.log" 2>&1
  py -3.12 soup_models.py --a "ck_soup/tmp_${OUT}a_050" --b "$3/$SUB" --alphas 0.3333333 --label "tmp_${OUT}b_" --out-dir ck_soup > "logs/soup_${OUT}_2.log" 2>&1
  mv "ck_soup/tmp_${OUT}b_033" "ck_soup/$OUT" && say "    created"; }

build2(){  # $1=type $2=arm A $3=arm B $4=output name (two-way 50/50)
  local SUB="$1" OUT="$4"
  [ -f "ck_soup/$OUT/checkpoint.pt" ] && { say "  already have ck_soup/$OUT"; return; }
  say "  two-way average -> ck_soup/$OUT"
  py -3.12 soup_models.py --a "$2/$SUB" --b "$3/$SUB" --alphas 0.5 --label "tmp_${OUT}_" --out-dir ck_soup > "logs/soup_${OUT}.log" 2>&1
  mv "ck_soup/tmp_${OUT}_050" "ck_soup/$OUT" && say "    created"; }

build_from_list(){  # $1=type $2=output name $3..=arm prefixes (any number, equal weight: iterative interpolation w=1/(i+1))
  local SUB="$1" OUT="$2"; shift 2
  local arms=("$@") n=${#@}
  [ -f "ck_soup/$OUT/checkpoint.pt" ] && { say "  already have ck_soup/$OUT"; return; }
  say "  ${n}-way average -> ck_soup/$OUT"
  local cur="${arms[0]}" i=1
  for a in "${arms[@]:1}"; do
    i=$((i+1))
    local w
    w=$(py -3.12 -c "print(1.0/$i)")
    py -3.12 soup_models.py --a "$cur/$SUB" --b "$a/$SUB" --alphas "$w" --label "tmp_${OUT}_$i" --out-dir ck_soup > "logs/soup_${OUT}_$i.log" 2>&1 || { say "    step $i failed"; return 1; }
    cur="ck_soup/tmp_${OUT}_${i}_$(py -3.12 -c "print(f'{round($w*100):03d}')")"
  done
  mv "$cur" "ck_soup/$OUT" && say "    created"; }

score20(){ local name="$1" tag="$2"
  [ -f "eval/noise20_${tag}_sig.json" ] && [ -f "eval/noise20_${tag}_nl.json" ] && { say "  already have $tag, skipping"; return; }
  say "  20-seed: $name"
  cp -f eval/eval_sft_summary.json eval/_keep_sig.json; cp -f eval/nl_eval_summary.json eval/_keep_nl.json
  py -3.12 eval_sft.py --models "$name" --seeds "$SEEDS20" --skip-humaneval > "logs/noise_sig_$tag.log" 2>&1; say "     sig rc=$?"
  cp -f eval/eval_sft_summary.json "eval/noise20_${tag}_sig.json"
  cp -f eval/eval_sft_generations.jsonl "eval/generations_noise20_${tag}_sig.jsonl" 2>/dev/null || true
  cp -f eval/_keep_sig.json eval/eval_sft_summary.json
  py -3.12 eval_nl.py --models "$name" --seeds "$SEEDS20" > "logs/noise_nl_$tag.log" 2>&1; say "     nl rc=$?"
  cp -f eval/nl_eval_summary.json "eval/noise20_${tag}_nl.json"
  cp -f eval/nl_eval_generations.jsonl "eval/generations_noise20_${tag}_nl.jsonl" 2>/dev/null || true
  cp -f eval/_keep_nl.json eval/nl_eval_summary.json
  say "     main summary restored: $(py -3.12 -c "import json;print(len(json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))))") entries"; }

say "########## round 4 start ##########"
say "== (1) top-two soup (s42 + s44; note the evaluation-selection bias) =="
build2 best_val ckpt_g9 ckpt_g9_s44 sG9top2bv
score20 "SOUP-G9top2-bestval" g9top2

say "== (2) add 5:1 seeds 45 / 46 (no selection bias) =="
train data_mix_b147 ckpt_g9_s45 45
eval_arm g9s45 "G9-g5to1-s45-final,G9-g5to1-s45-bestval"
train data_mix_b147 ckpt_g9_s46 46
eval_arm g9s46 "G9-g5to1-s46-final,G9-g5to1-s46-bestval"
score20 "G9-g5to1-s45-bestval" g9s45
score20 "G9-g5to1-s46-bestval" g9s46

say "== (3) five-way soup (s42..s46, equal weight) =="
build_from_list best_val sG9x5bv ckpt_g9 ckpt_g9_s43 ckpt_g9_s44 ckpt_g9_s45 ckpt_g9_s46
score20 "SOUP-G9x5-bestval" g9x5

say "== summary (20-seed precision protocol) =="
py -3.12 - <<'PY'
import json
def load(tag):
    try:
        s=json.load(open(f"eval/noise20_{tag}_sig.json",encoding="utf-8")); s=s if isinstance(s,list) else s["models"]
        n=json.load(open(f"eval/noise20_{tag}_nl.json",encoding="utf-8")); n=n if isinstance(n,list) else n["models"]
    except Exception: return None
    def v(ms):
        t=ms[0]["tasks"]; a=sum(int(x.split("/")[0]) for x in t.values()); b=sum(int(x.split("/")[1]) for x in t.values()); return a/b*55
    return v(n), v(s)
order=[("G9-g5to1-bestval(5:1 s42)","g9bv"),("G9-g5to1-s43-bestval","g9s43"),("G9-g5to1-s44-bestval","g9s44"),
       ("G9-g5to1-s45-bestval","g9s45"),("G9-g5to1-s46-bestval","g9s46"),
       ("SOUP-G9top2(top-two soup)","g9top2"),("SOUP-G9x3(three-way soup)","g9x3bv"),("SOUP-G9x5(five-way soup)","g9x5"),
       ("-[control] SOUP-G8x3(3:1 three-way soup)","soup3"),("-[control] G8-g3to1-bestval(3:1 s42)","g8bv")]
print(f"{'model':<34}{'NL':>7}{'sig':>7}{'total':>7}")
rows=[]
for label,tag in order:
    r=load(tag)
    if r is None: print(f"{label:<34}{'(not measured)':>21}"); continue
    print(f"{label:<34}{r[0]:>7.1f}{r[1]:>7.1f}{r[0]+r[1]:>7.1f}")
    rows.append((label,r[0]+r[1]))
vals=[t for l,t in rows if 'G9-g5to1' in l]
if vals:
    import statistics as st
    print(f"\n5:1 recipe seed mean {st.mean(vals):.1f} (n={len(vals)}, range {max(vals)-min(vals):.1f})")
    print(f"of which top-two soup = {dict(rows).get('SOUP-G9top2(top-two soup)', float('nan')):.1f}; five-way soup = {dict(rows).get('SOUP-G9x5(five-way soup)', float('nan')):.1f}")
PY
say "########## round 4 done ##########"
