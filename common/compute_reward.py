# pip install math-verify
from math_verify import parse, verify
import re 
import os

## vlm as judger for conditional reward 
from openai import OpenAI
import requests
import random

## llm as judge for acc reward
llm_openai_api_key = "EMPTY"
llm_openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://127.0.0.1:12347/v1"),
]
print("!!!! [debug] llm as a judge base list", llm_openai_api_base_list)

llm_client_list = []
llm_model_name_list = []
for api_base in llm_openai_api_base_list:
    client = OpenAI(
        api_key=llm_openai_api_key,
        base_url=api_base,
    )
    llm_client_list.append(client)

    response = requests.get(f"{api_base}/models")
    models = response.json()
    llm_model_name_list.append(models['data'][0]['id'])

def do_verify(nsol, b):
    res = 0.0 
    try:
        a = parse(nsol)
        if len(b)>1 and (b[1] in 'ABCDEFGHIJK'):
            res = float(nsol[len("\\boxed{"):].startswith(b[1]))
        else:
            # print(f"debug parsed: {a} from {nsol} and {b}")
            if len(a)==0: res = -1.0 
            else: res = float(verify(a, b))
    except: 
        print(f"!!!! [debug] {nsol} parsing exception")
        res = -1.0 
    return res 

ans_indicator = "answer is"
endstr = "Now everything looks fine. Solution finished."
def normalize_answer(answer):
    if answer is None: return answer
    if 'dfrac' in answer: answer = answer.replace("dfrac", "frac")
    # if '%' in answer: answer = answer.replace(r'\%',"").replace('%',"")
    if 'text' in answer: answer = answer.replace("\\text","")
    if "\\varnothing" in answer: answer = answer.replace("\\varnothing","\\emptyset")
    if "minutes" in answer: answer = answer.replace("minutes","")
    if "cm" in answer: answer = answer.replace("cm","")
    # if "^\\circ" in answer: answer = answer.replace("^\\circ","")
    # if "a.m." in answer: answer = answer.replace("a.m.","")
    return answer 

def handle_boxed(sol, gt, eostoken, format_type, requires_box=False):
    # print(sol)
    # print('!!!! debug', gt)
    norepeat = None
    usefmt = None
    res = 0.0
    # endstr and eos token
    index = sol.find(endstr)
    num_end = len(eostoken)
    
    if index>-1: 
        remains = sol[index+len(endstr)+num_end:]
        if len(remains)>0: 
            norepeat = False 
        else: norepeat = True 
    if not (norepeat is False):
        if format_type in ["confidence"]:
            if not ("<confidence>" in sol and "</confidence>" in sol):
                usefmt = False
            else: 
                count = sol.count("<confidence>")
                if count>5: usefmt = False 
                else: usefmt = True 
        elif format_type in ["wait"]:
            tmps = sol.lower()
            usefmt = False 
            if "wait" in tmps or "alternatively" in tmps:
                usefmt = True 
            
        # elif format_type in ["nocode"]:
        #     if "```python" in sol:
        #         usefmt = False 
        #     else: usefmt = True 
    
    if (norepeat is False): 
        pass # no need for correctness 
    else: 
        flag = True 
        gt = normalize_answer(gt)
        try:
            if "\\boxed" in gt: 
                b = parse(gt)
                # print('!!!! debug gt parse', gt, b)
            else:
                b = parse(f"\\boxed{{{gt}}}")
        except Exception as e:
            print(f"!!!! [debug] {gt} parsing exception")
            res = -1.0
            flag = False 
        if flag:
            if len(b)==0: res = -1.0 
            else: 
                if requires_box:
                    boxed_index = sol.rindex("boxed")
                    if boxed_index==-1: res = 0.0 
                    else:
                        nsol = '\\'+sol[boxed_index:]
                        res = do_verify(normalize_answer(nsol), b)
                else: 
                    flag = False 
                    
                    for indicator in ["\\boxed", "<answer>", "Answer:"]:
                        if indicator in sol:
                            if indicator == "<answer>":
                                found = re.search("<answer>(.*?)</answer>", sol)
                                if found:
                                    nsol = f"\\boxed{{{found.group(1)}}}"
                                else: continue 
                            elif indicator == "Answer:":
                                tmp = sol.split(indicator)
                                if len(tmp)>0: tmp = tmp[-1].strip() # .split(eostoken)[0].strip()
                                else: continue 
                                nsol = f"\\boxed{{{tmp}}}"
                            else: 
                                boxed_index = sol.rindex(indicator)
                                pred = sol[boxed_index:].strip()
                                nsol = pred
                            res = do_verify(normalize_answer(nsol), b)
                            if res > 0.99: 
                                flag = True 
                        if flag: 
                            break
                    # print("extracted sol", nsol)
                    if not flag:
                        nsol = sol 
                        res = do_verify(normalize_answer(nsol), b)
        
                    
    return norepeat, usefmt, res 

############# LLM as Judge #############
def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt


def llm_as_judge(question, sol, gt):
    full_prompt = get_prompt(sol, gt, question)

    client_idx = random.randint(0, len(llm_client_list) - 1)
    client = llm_client_list[client_idx]
    model_name = llm_model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    # print(response)
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0    

    # print(f"[DEBUG] LLM as Judge:\nQuestion: {question}\nPred_Ans: {sol}\nGT_Ans: {gt}\nJudgement: {acc_reward}")
    
    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(sol) >= 1000:
        acc_reward = -1.0       # 0.0
        # is_format_error = True
    return acc_reward

def rule_reward(question, sol, gt, eostoken, format_type, requires_box, *args):
    # valid = eos & boxed 
    error_info = None 
    valid = True 
    if eostoken not in sol and "<|endoftext|>" not in sol: 
        valid = False
        error_info = "No eos."
    elif requires_box and "boxed" not in sol:
        valid = False 
        error_info = "No valid boxed."
    elif sol.lower().count("wait")>5:
        valid = False
        error_info = "too many waits"
    ############ this is only for debugging 
    # if format_type=='wait':
    # tmps = sol.lower() 
    # if "wait" not in tmps and "alternatively" not in tmps:
    #     valid = False 
    
    # formats and correctness 
    norepeat = None
    usefmt = None
    res = 0.0
    # directly making no-boxed and no-eos as invalid seems unnecessary and harmful:
    # the model needs to understand these are unacceptable
    # if not valid: 
    #     pass 
    # else: 
    if isinstance(gt, list):
        gt = [xx.lower() for xx in gt]
        has_percent = None
        for xx in gt:
            if "%" in xx: 
                has_percent = xx.replace('%','')
                break
        if has_percent is not None and has_percent not in gt:
            gt.insert(0, has_percent)
    else:
        gt = [str(gt).lower()]
    tmpsol = sol.lower()
    # special_char = {'%'}
    for ans in gt:
        norepeat, usefmt, res = handle_boxed(tmpsol, ans, eostoken, format_type, requires_box=requires_box)
        if res>0.5:
            break
        else:
            ##### llm as judge #####
            # split \boxed{}
            pure_ans = ans 
            if "\\boxed" in ans:
                pure_ans = ans[ans.index("\\boxed{")+len("\\boxed{"):ans.index("}")]

            boxed_index = tmpsol.rfind("boxed")
            if boxed_index==-1: continue
            else:
                nsol = '\\'+tmpsol[boxed_index:]
                try:
                    pure_sol = nsol[nsol.index("\\boxed{")+len("\\boxed{"):nsol.index("}")]
                except:
                    # 没有boxed，可能报错ValueError: substring not found
                    pure_sol = nsol
            
            # get the question
            # llm as judge 
            res = llm_as_judge(question, pure_sol, pure_ans)
            if res>0.5:
                break

    return valid, norepeat, usefmt, error_info, res 
