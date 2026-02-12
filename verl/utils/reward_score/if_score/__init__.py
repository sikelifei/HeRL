from verl.utils.reward_score.if_score.ifeval.evaluation_lib import test_ifeval_strict
from verl.utils.reward_score.if_score.ifbench.evaluation_lib import test_ifbench_strict
from verl.utils.reward_score.if_score.muldimif.evaluation import test_muldimif_strict

from verl.utils.reward_score.if_score.ifeval import instructions_registry as ifeval_registry
from verl.utils.reward_score.if_score.ifbench import instructions_registry as ifbench_registry

import json

def _build_instruction_details(ground_truth, is_following_list, registry):
    details = []
    instruction_ids = ground_truth.get("instruction_id_list") or []
    kwargs_list = ground_truth.get("kwargs") or []
    for idx, instruction_id in enumerate(instruction_ids):
        kwargs = {}
        description = instruction_id
        try:
            instruction_cls = registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict):
                kwargs = {k: v for k, v in kwargs_list[idx].items() if v is not None}
            description = instruction.build_description(**kwargs)
        except Exception:
            pass
        details.append({
            "instruction_id": instruction_id,
            "description": description,
            "kwargs": kwargs,
            "is_following": bool(is_following_list[idx]),
        })
    return details

def _build_muldimif_details(ground_truth, is_following_list):
    constraints = ground_truth.get("instruction_id_list") or []
    details = []
    for constraint, flag in zip(constraints, is_following_list):
        section = constraint[0] if len(constraint) > 0 else ""
        sub_section = constraint[1] if len(constraint) > 1 else ""
        requirement = constraint[-1] if constraint else ""
        if isinstance(requirement, dict):
            requirement_text = json.dumps(requirement, ensure_ascii=False)
        else:
            requirement_text = str(requirement)
        description = f"{section} - {sub_section}: {requirement_text}".strip(" -:")
        details.append({
            "instruction_id": constraint,
            "description": description,
            "kwargs": requirement if isinstance(requirement, dict) else None,
            "is_following": bool(flag),
        })
    return details


def contain_repeated_text(text):
    detected_text = text[-160:]
    test_spilt = text.split(detected_text)

    if len(test_spilt) > 3:
        return True
    else:
        return False

def _sanitize_kwargs_list(ground_truth):
    kwargs_list = ground_truth.get("kwargs") or []
    sanitized_list = []
    for kwargs in kwargs_list:
        if isinstance(kwargs, dict):
            sanitized_list.append({k: v for k, v in kwargs.items() if v is not None})
        else:
            sanitized_list.append({})
    return sanitized_list


def compute_score(model_output: str, ground_truth, type) -> bool:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    kwargs_list = _sanitize_kwargs_list(ground_truth)
    ground_truth["kwargs"] = kwargs_list

    try:
        if type == "ifeval":
            sparse_score, dense_score, is_following_list = test_ifeval_strict(model_output, ground_truth)
            constraint_results = _build_instruction_details(ground_truth, is_following_list, ifeval_registry)
        elif type == "ifbench":
            sparse_score, dense_score, is_following_list = test_ifbench_strict(model_output, ground_truth)
            constraint_results = _build_instruction_details(ground_truth, is_following_list, ifbench_registry)
        elif type == "muldimif":
            sparse_score, dense_score, is_following_list = test_muldimif_strict(model_output, ground_truth)
            constraint_results = _build_muldimif_details(ground_truth, is_following_list)
        else:
            raise Exception("Invalid type")
        #print("sparse_score:", sparse_score, "dense_score:", dense_score)
        if contain_repeated_text(model_output):
            return {"score": dense_score, "sparse": sparse_score, "dense": 0, "is_following_list": None, "constraint_results": None}
        else:
            return {"score": dense_score, "sparse": sparse_score, "dense": dense_score, "is_following_list": is_following_list, "constraint_results": constraint_results}
    except:
        print("rule verify timeout!")
        # print("model_output:", model_output[:20])
        return {"score": 0, "sparse": 0, "dense": 0, "is_following_list": None, "constraint_results": None}
    
