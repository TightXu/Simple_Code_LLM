#!/usr/bin/env bash
# Round 2 (user authorized on 09-14 02:0x to "do what you think is best"):
#   background: the overnight chain proved 5-seed single-arm readings are drowned in noise (same recipe seed 42/43/44 = 81/75/73;
#        whereas 20-seed precision measurement shows the champion seed42 is really only 76.0 and the three-way soup is 79.8).
#   -> so this round: (1) build a three-way soup for the 1.5:1 and 5:1 recipes too; (2) use the same set of 20 evaluation seeds
#     to re-measure the key candidates, yielding a mutually comparable precision board (the 5-seed and 20-seed boards cannot be mixed, the seed sets differ).
# Order: build soups (CPU) -> numerical check -> 20-seed evaluation one by one (isolated: back up main summary -> run -> save aside -> restore) -> summary
set -u
cd "$(dirname "$0")/.." || exit 1
exec > >(tee -a logs/chain_recipe_soup.log) 2>&1
say(){ echo "[$(date '+%F %T')] $*"; }
SEEDS20="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"

build3(){  # $1=checkpoint type (final|best_val)  $2=arm prefixes (three, space-separated)  $3=output dir name
  local SUB="$1" OUT="$3" A B C
  set -- $2; A="$1"; B="$2"; C="$3"
  [ -f "ck_soup/$OUT/checkpoint.pt" ] && { say "  already have ck_soup/$OUT, skipping"; return; }
  say "  three-way average ($SUB) -> ck_soup/$OUT   [$A + $B + $C]"
  py -3.12 soup_models.py --a "$A/$SUB" --b "$B/$SUB" --alphas 0.5 --label "tmp_${OUT}a_" --out-dir ck_soup > "logs/soup_${OUT}_1.log" 2>&1 || { say "    step1 failed"; return 1; }
  py -3.12 soup_models.py --a "ck_soup/tmp_${OUT}a_050" --b "$C/$SUB" --alphas 0.3333333 --label "tmp_${OUT}b_" --out-dir ck_soup > "logs/soup_${OUT}_2.log" 2>&1 || { say "    step2 failed"; return 1; }
  mv "ck_soup/tmp_${OUT}b_033" "ck_soup/$OUT" && say "    created ck_soup/$OUT"; }

verify3(){  # $1=type $2=arm prefixes (three) $3=output dir -- reconcile against the true average
  local SUB="$1" OUT="$3"; set -- $2
  py -3.12 - "$1" "$2" "$3" "$SUB" "$OUT" <<'PY'
import sys, torch
A,B,C,SUB,OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
K=["blocks.0.attn.qkv.weight","embed.weight"]
def g(p):
    ck=torch.load(p, map_location="cpu", weights_only=False)
    sd={k.replace("_orig_mod.",""):v for k,v in ck["model"].items()}
    return {k: sd[k].float() for k in K}
a = g(f"{A}/{SUB}/checkpoint.pt"); b = g(f"{B}/{SUB}/checkpoint.pt")
c = g(f"{C}/{SUB}/checkpoint.pt"); s = g(f"ck_soup/{OUT}/checkpoint.pt")
for k in K:
    want=(a[k]+b[k]+c[k])/3
    dev=(s[k]-want).abs().max().item()
    print(f"    {k}: deviation from the true average {dev:.2e} (magnitude {want.abs().max().item():.2e}) -> {'match' if dev<1e-3*max(want.abs().max().item(),1e-6) else 'large deviation!'}")
PY
}

score20(){  # $1=model name $2=short tag
  local name="$1" tag="$2"
  [ -f "eval/noise20_${tag}_sig.json" ] && [ -f "eval/noise20_${tag}_nl.json" ] && { say "  already have 20-seed results for $tag, skipping"; return; }
  say "  20-seed evaluation: $name"
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
  say "     main summary restored: sig $(py -3.12 -c "import json;print(len(json.load(open('eval/eval_sft_summary.json',encoding='utf-8'))))") entries"; }

say "########## round 2 start ##########"
say "== build recipe-level soups =="
build3 final    "ckpt_g5 ckpt_g5_s43 ckpt_g5_s44" sG5x3fi
build3 best_val "ckpt_g9 ckpt_g9_s43 ckpt_g9_s44" sG9x3bv
build3 best_val "ckpt_g5 ckpt_g5_s43 ckpt_g5_s44" sG5x3bv

say "== numerical check (must equal the true average) =="
verify3 final    "ckpt_g5 ckpt_g5_s43 ckpt_g5_s44" sG5x3fi
verify3 best_val "ckpt_g9 ckpt_g9_s43 ckpt_g9_s44" sG9x3bv

say "== 20-seed precision evaluation (same seed set -> mutually comparable) =="
score20 "SOUP-G5x3-final"   g5x3fi
score20 "SOUP-G9x3-bestval" g9x3bv
score20 "SOUP-G9G5-030"     g9g5_030
score20 "G9-g5to1-bestval"  g9bv

say "== summary: 20-seed precision board (scaled to the 5-seed protocol x0.25 for comparison with the old board) =="
py -3.12 - <<'PY'
import json, glob, os
rows=[]
for f in sorted(glob.glob("eval/noise20_*_sig.json")):
    tag=os.path.basename(f)[len("noise20_"):-len("_sig.json")]
    try:
        s=json.load(open(f,encoding="utf-8")); s=s if isinstance(s,list) else s["models"]
        n=json.load(open(f"eval/noise20_{tag}_nl.json",encoding="utf-8")); n=n if isinstance(n,list) else n["models"]
    except Exception as e:
        print(f"  {tag}: read failed {e}"); continue
    def v(ms):
        t=ms[0]["tasks"]; a=sum(int(x.split("/")[0]) for x in t.values()); b=sum(int(x.split("/")[1]) for x in t.values())
        return a,b,ms[0]["model"]
    sa,sb,sname=v(s); na,nb,_=v(n)
    rows.append((sname, na/nb*55, sa/sb*55, na/nb*55+sa/sb*55, na/nb*100, sa/sb*100, nb))
rows.sort(key=lambda r:-r[3])
print(f"{'model':<26}{'NL(scaled)':>10}{'sig(scaled)':>11}{'total':>8}{'NL%':>8}{'sig%':>8}{'seeds':>7}")
for r in rows:
    print(f"{r[0]:<26}{r[1]:>10.1f}{r[2]:>11.1f}{r[3]:>8.1f}{r[4]:>7.1f}%{r[5]:>7.1f}%{r[6]:>7}")
print("\n(scaled: 20-seed percentage x 55 = expected total if 5 seeds were used, for side-by-side comparison with the old 5-seed board)")
PY
say "########## round 2 done ##########"
