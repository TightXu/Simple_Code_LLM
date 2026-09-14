#!/usr/bin/env python3
"""
HumanEval 评测结果汇总 + 抽样审核 (nofilter@18B vs v2@18B)
==========================================================
评测完成后运行, 输出:
- 参考通过率汇总 (两模型 × 163 题 × 10 seed)
- 失败原因分布 (判定类别)
- 抽样生成内容 (前几题) 供人工审核

用法: py -3.12 summarize_humaneval.py <generations.jsonl 路径> [模型2路径]
"""
import json, sys, os
from collections import defaultdict, Counter

def main():
    if len(sys.argv) < 2:
        print("用法: summarize_humaneval.py <generations.jsonl>")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"[ERR] {path} 不存在")
        return
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    print(f'总生成: {len(rows)} 条')
    if not rows:
        print("(空)")
        return

    # 按模型分组
    by_model = defaultdict(list)
    for r in rows:
        mk = 'nofilter@18B' if 'nofilter_full' in r['model'] else ('v2@18B' if 'v2_18b' in r['model'] else r['model'].split('/')[-2])
        by_model[mk].append(r)

    print("\n=== 参考通过率 (per model) ===")
    for mk, rs in sorted(by_model.items()):
        passed = sum(1 for r in rs if r['reference_pass'])
        # 按算法(HumanEval/N)分组
        alg_stats = Counter()
        for r in rs:
            alg_stats[r['algorithm']] += 1
        print(f'  {mk}: {passed}/{len(rs)} pass ({passed/len(rs)*100:.1f}%)')
        # 每个任务通过情况 (任务级: 10 seed 中任意 pass 则该任务 pass)
        task_pass = defaultdict(int)   # task -> passed seeds count
        for r in rs:
            task_pass[r['algorithm']] += int(r['reference_pass'])
        total_tasks = len(task_pass)
        tasks_any_pass = sum(1 for k, v in task_pass.items() if v > 0)
        print(f'    tasks: {tasks_any_pass}/{total_tasks} 有任何 seed pass')

    print("\n=== 失败原因分布 (per model) ===")
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

    print("\n=== 抽样生成 (前 6 条, 供人工审核) ===")
    for mk, rs in sorted(by_model.items()):
        print(f'\n---- {mk} ----')
        for r in rs[:3]:
            g = r['generation']
            print(f'[{r["algorithm"]} seed={r["seed"]} pass={r["reference_pass"]}]')
            print('  ' + g[:120].replace('\n', '\n  '))
            print()

    print("\n=== 建议 ===")
    print("参考 pass 率仅作参考; 决定性结果是人工审核 (0/1/2 分)。")

if __name__ == '__main__':
    main()
