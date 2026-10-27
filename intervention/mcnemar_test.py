import os.path as path
import json
import argparse
import numpy as np
import statsmodels.stats.contingency_tables as ssc

def load_score_results(results_file):
    with open(results_file, "r") as f:
        if results_file.endswith(".jsonl"):
            results = [json.loads(line) for line in f]
        else:
            results = json.load(f)
        print(f"load results from {results_file}")
    # filter out 'has_toolcall' is False
    results = [item for item in results if item["has_toolcall"]]
    # 按"qid"的值进行排序
    results = sorted(results, key=lambda x: x["qid"])
    return results

def get_score(reuslts_file):
    results = load_score_results(reuslts_file)
    aa = [float(item["score"]) for item in results]
    return np.mean(aa)

def get_paired_score(group_a, group_b):
    a_score = load_score_results(group_a)
    b_score = load_score_results(group_b)
    # 严格的一一对应检查
    assert len(a_score) == len(b_score), f"Length mismatch: {len(a_score)} vs {len(b_score)}"

    aa = []; bb = []
    for a, b in zip(a_score, b_score):
        assert a['qid'] == b['qid']
        aa.append(float(a['score']))
        bb.append(float(b['score']))
    return aa, bb

def mcnemar_test(aa, bb):
    aa = np.array(aa)
    bb = np.array(bb)
    table = np.array([[0., 0.], [0., 0.]])
    table[0, 0] = ((1 - aa) * (1 - bb)).sum()
    table[0, 1] = ((1 - aa) * bb).sum()
    table[1, 0] = (aa * (1 - bb)).sum()
    table[1, 1] = (aa * bb).sum()
    result = ssc.mcnemar(table, exact=True, correction=True)
    return result.statistic, result.pvalue

def get_average_treatment_effect(group_a_file, group_b_file):
    aa, bb = get_paired_score(group_a_file, group_b_file)
    base = np.mean(aa)
    ate = np.mean(bb) - np.mean(aa)
    _, pvalue = mcnemar_test(aa, bb)
    return base, ate, pvalue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_a', type=str, required=True, help='Path to baseline results (relative to the working directory or absolute)')
    parser.add_argument('--group_b', type=str, required=True, help='Path to intervention results (relative to the working directory or absolute)')
    args = parser.parse_args()


    aa, bb = get_paired_score(args.group_a, args.group_b)
    print(np.mean(aa))
    print(np.mean(bb))

    base, ate, pvalue = get_average_treatment_effect(args.group_a, args.group_b)
    print("--" * 20)
    print(f"Base: {base:.4f}, ATE: {ate:.4f}, p-value: {pvalue:.4f}")
    print("--" * 20)
