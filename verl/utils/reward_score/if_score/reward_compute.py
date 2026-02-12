from verl.utils.reward_score.if_score import compute_score as if_score_compute

SUPPORTED = {"ifeval", "ifbench", "muldimif"}

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    if data_source not in SUPPORTED:
        raise ValueError(f"unsupported data_source={data_source}")
    
    # print("[RM DEBUG] data_source", data_source)
    # print("[RM DEBUG] solution_str", solution_str[:200])
    # print("[RM DEBUG] ground_truth", ground_truth)
    # print("[RM DEBUG] extra_info keys", extra_info.keys())

    return if_score_compute(solution_str, ground_truth, data_source)

