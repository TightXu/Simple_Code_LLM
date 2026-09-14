#!/usr/bin/env python3
"""
HumanEval evaluation summary + sampled review (nofilter@18B vs v2@18B)
==========================================================
Run after the evaluation finishes; prints:
- reference pass-rate summary (2 models x 163 tasks x 10 seeds)
- failure-reason breakdown (judge categories)
- sampled generations (first few tasks) for human review

Usage: py -3.12 summarize_humaneval.py <generations.jsonl path> [second model path]
"""
import json, sys, os
from collections import defaultdict, Counter

def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_humaneval.py <generations.jsonl>")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"[ERR] {path} does not exist")
        return
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    print(f'Total: {len(rows)} generations')
    if not rows:
        print("(empty)")
        return

    # group by model
    by_model = defaultdict(list)
    for r in rows:
        mk = 'nofilter@18B' if 'nofilter_full' in r['model'] else ('v2@18B' if 'v2_18b' in r['model'] else r['model'].split('/')[-2])
        by_model[mk].append(r)

    print("\n=== Reference pass rate (per model) ===")
    for mk, rs in sorted(by_model.items()):
        passed = sum(1 for r in rs if r['reference_pass'])
        # group by algorithm (HumanEval/N)
        alg_stats = Counter()
        for r in rs:
            alg_stats[r['algorithm']] += 1
        print(f'  {mk}: {passed}/{len(rs)} pass ({passed/len(rs)*100:.1f}%)')
        # per-task pass status (task level: a task passes if any of its 10 seeds passes)
        task_pass = defaultdict(int)   # task -> passed seeds count
        for r in rs:
            task_pass[r['algorithm']] += int(r['reference_pass'])
        total_tasks = len(task_pass)
        tasks_any_pass = sum(1 for k, v in task_pass.items() if v > 0)
        print(f'    tasks: {tasks_any_pass}/{total_tasks} with at least one passing seed')

    print("\n=== Failure-reason breakdown (per model) ===")
    for mk, rs in sorted(by_model.items()):
        cats = Counter()
        for r in rs:
            reason = r.get('reason', '')
            if reason.startswith('impl: MISSING'): cats['MISSING_struct'] += 1
            elif reason.startswith('impl: BANNED'): cats['BANNED_cheat'] += 1
            elif reason.startswith('exec:'): cats['exec_fail'] += 1
            elif 'no function body' in reason: cats['no_fn'] += 1
            elif reason == 'pass': cats['PASS'] += 1
            else: cats['other:' + reason[:20]] += 1
        print(f'  {mk}: {dict(cats)}')

    print("\n=== Sampled generations (first 6, for human review) ===")
    for mk, rs in sorted(by_model.items()):
        print(f'\n---- {mk} ----')
        for r in rs[:3]:
            g = r['generation']
            print(f'[{r["algorithm"]} seed={r["seed"]} pass={r["reference_pass"]}]')
            print('  ' + g[:120].replace('\n', '\n  '))
            print()

    print("\n=== Recommendation ===")
    print("The reference pass rate is advisory only; the decisive result is human review (scored 0/1/2).")

if __name__ == '__main__':
    main()
