# -*- coding: utf8 -*-
import openai
from http import HTTPStatus
import argparse
import os
import base64
import json
import time
import random
from tqdm import tqdm
from datetime import datetime
from time import sleep
import re
from typing import Optional
from utils import Client, openai_chat
from concurrent.futures import ThreadPoolExecutor
import ast

ds_collections = {
    'CMMU-Base':{
        'root': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu',
        'question': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu/chemistry_conv.jsonl',
    },
    'CMMU-Thinking':{
        'root': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu',
        'question': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu/chemistry_conv.jsonl',
    }
}

PROMPT_EXTRACT_CHOCIE_ZH = '''你是一个专业的答案提取器，你的任务是根据给定的多项选择题问题和回答，提取出选择的答案。如果无法从回答中提取出明确的答案，则输出选项“Z”。
要求：
1、请从回答中提取出选择的答案（字母，可能不止一个），不允许改变回答的意思；
2、如果无法从回答中提取出明确的答案，则输出选项“Z”；
3、请输出JSON格式，包含两个字段：
    - "extracted_answer"：包含提取出的答案字母的列表，如果无法提取则为["Z"]；
    - "status"：提取结果的状态，若提取成功则为 "success"，否则为 "failure"；
JSON输出示例整体如下：
{"extracted_answer":["字母1","字母2,...], "status" :"success / failure"}

以下是提供的问题和回答。
问题：
```
{question}
```
回答：
```
{response}
```

请以"答案提取:"开头，后跟JSON输出。
'''

PROMPT_EXTRACT_CHOCIE='''You are a professional answer extractor. Your task is to extract the selected answer(s) based on the given multiple-choice question and response. If it is not possible to extract a clear answer from the response, output option "Z."

Requirements:
1. Extract the answer(s) from the response (letters, potentially more than one) without changing the meaning of the response.
2. If a clear answer cannot be extracted from the response, output option "Z."
3. Output in JSON format with two fields:
   - "extracted_answer": A list of the extracted answer letters; if extraction is not possible, it should be ["Z"].
   - "status": The status of the extraction result. If successful, the status should be "success," otherwise "failure."

Example JSON output:
```
{"extracted_answer":["letter1","letter2,..."], "status":"success / failure"}
```

Here is the provided question and response:
Question:
```
{question}
```
Response:
```
{response}
```

Please begin the response with "Answer Extraction:" followed by the JSON output.
'''

PROMPT_EXTRACT_BLANK_ZH = '''你是一个专业的答案提取器，你的任务是根据给定的问题和回复分别提取每个子问题的回答，如果某个子问题无法提取答案，请写入""。
要求：
1、我将提供问题和回答，子问题以列表的形式给出，请从回答中提取出选择的答案（字母，可能不止一个），不允许改变回答的意思；
2、如果缺少某个子问题的回答，请写入""，即空字符串；
3、以字典列表格式输出，列表顺序与子问题列表相同，字典包含三个字段：
    - "sub_question"： 子问题；
    - "extracted_answer"：对应子问题的回答；
    - "status"：提取结果的状态，若提取成功则为 "success"，否则为 "failure"；
JSON输出示例整体如下：
[{"sub_question":"sub-question1","extracted_answer":"answer for sub-question1", "status" :"success / failure"},
{"sub_question":"sub-question2","extracted_answer":"answer for sub-question2", "status" :"success / failure"},
...]

以下是提供的问题和回答。
问题：
```
{question}
```
回答：
```
{response}
```

请以"答案提取:"开头，后跟JSON输出。
''' 
PROMPT_EXTRACT_BLANK='''You are a professional answer extractor. Your task is to extract the answers for each sub-question from the given question and response. If an answer to a sub-question cannot be extracted, please write an empty string.

Requirements:
1. I will provide the question and the response, with the sub-questions listed. You need to extract the corresponding answers (letters, possibly multiple) from the response, without changing the meaning of the response.
2. If the answer for any sub-question is missing, please write an empty string ("").
3. Output the results in the form of a list of dictionaries, with the list order matching that of the sub-questions. Each dictionary should include three fields:
    - "sub_question": The sub-question;
    - "extracted_answer": The extracted answer for the sub-question;
    - "status": The status of the extraction, where "success" indicates a successful extraction, and "failure" indicates an unsuccessful extraction.

Example JSON output format:
```
[{"sub_question":"sub-question1","extracted_answer":"answer for sub-question1", "status" :"success / failure"},
{"sub_question":"sub-question2","extracted_answer":"answer for sub-question2", "status" :"success / failure"},
...]
```

The following is the provided question and response:
Question:
```
{question}
```
Response:
```
{response}
```

Please begin with "Answer Extraction:" followed by the JSON output.
'''
PROMPT_EVAL_BLANK_ZH = '''你是一个专业的评测助手，你的任务是根据提供的每个问题和对应的标准答案判断AI模型的预测是否正确.
要求：
1、你将被提供问题，对应的标准答案和AI模型预测，这些内容将以字典列表的形式给出，每一个字典对应题目的一个子问题；
2、对于每个子问题，分别根据标准答案，评测对应的AI模型的预测是否正确，正确输出1，错误输出0, 如果缺少预测，也输出0；
3、在进行评估时，AI模型的预测的格式或者表达方式不需要与标准答案的完全一致，但模型预测的要素必须齐全，含义或核心内容必须与标准答案相同；
4、以字典列表格式输出，每个字典相比提供的字典多两个字段：
    - "analysis": 分析AI模型的预测是否正确，作为score的依据；
    - "score"：表示AI模型的预测是否正确，正确赋值1，错误为0。
JSON输出示例整体如下：
[{"sub_question":"sub-question 1","groundtruth_answer":"groundtruth answer for sub-question 1", "AI prediction" :"AI model's prediction for sub-question 1", "analysis":"Explain whether the AI model's prediction is correct", "score":0/1},
{"sub_question":"sub-question2","groundtruth_answer":"groundtruth answer for sub-question1", "AI prediction" :"AI model's prediction for sub-question 1","analysis":"Explain whether the AI model's prediction is correct", "score":0/1},
...]

以下是提供的题干，子问题，对应的标准答案和AI模型预测。
题干：
```
{main_question}
```
子问题，对应的标准答案和AI模型预测：
```
{sub_ques_gt_pred}
```

请以"分数:"开头，后跟JSON输出。
'''

PROMPT_EVAL_BLANK = '''You are a professional evaluation assistant. Your task is to assess whether the AI model's predictions are correct based on the provided questions and corresponding ground truth answers.

Requirements:
1. You will be provided with questions, their corresponding ground truth answers, and AI model predictions. These will be given as a list of dictionaries, each dictionary corresponding to a sub-question;
2. For each sub-question, evaluate whether the AI model's prediction is correct based on the ground truth answer. Output 1 for correct, 0 for incorrect. If the prediction is missing, output 0 as well;
3. The format or expression of the AI model's prediction does not need to be identical to the ground truth answer, but the key elements of the prediction must be complete, and its meaning or core content must be the same as the ground truth answer;
4. Output the results as a list of dictionaries. Output the results as a list of dictionaries. Each dictionary should include two additional fields:
    - "analysis": A simple analysis explaining whether the AI model's prediction is correct, which serves as the basis for the score.
    - "score": Indicates whether the AI model's prediction is correct, with the value set to 1 for correct and 0 for incorrect.
Example of JSON output:
```
[
    {"sub_question":"sub-question 1", "groundtruth_answer":"groundtruth answer for sub-question 1", "AI prediction":"AI model's prediction for sub-question 1", "analysis":"Explain whether the AI model's prediction is correct", "score":0/1},
    {"sub_question":"sub-question 2", "groundtruth_answer":"groundtruth answer for sub-question 2", "AI prediction":"AI model's prediction for sub-question 2", "analysis":"Explain whether the AI model's prediction is correct", "score":0/1},
    ...
]
```

The main question of all sub-questions:
```
{main_question}
```

The provided Sub-questions, corresponding ground truth answers, and AI model predictions:
```
{sub_ques_gt_pred}
```

Please begin with with "Score:" followed by the JSON output.
'''

def list2id_dict(data_list):
    result = {}
    for data_item in data_list:
        if data_item["id"] in result:
            print(f"\nDuplicate id: {data_item['id']}\nPrevious record:\n {result[data_item['id']]}\nCurrent record:\n {data_item}\n")
        else:
            result[data_item['id']] = data_item

    return result  

def extract_multi_choice_answer(question:str, response:str):
    
    prompt  = PROMPT_EXTRACT_CHOCIE.replace("{question}",question).replace("{response}",response)
    # print(prompt)
    call_out = openai_chat(Client,"gpt-4o",question=prompt)
    print(call_out)
    answer = [] # multi_chocie
    if call_out:
        try:
            json_part = call_out.split("Answer Extraction:")[-1].strip().strip("`")
            if json_part.startswith("json"):
                json_part = json_part[len("json"):].strip()
            extracted_data = json.loads(json_part)
            answer = extracted_data['extracted_answer']
        except  Exception as e:
            print(e)
            # print(json_part)
    else:
        print(question)
    return answer

def extract_multi_blank_answer(main_question:str,
                                sub_questions:list,
                                response:list):
    question_dict = {
        "main_question":main_question,
        "sub_questions":sub_questions,
    }
    prompt  = PROMPT_EXTRACT_BLANK.replace("{question}",json.dumps(question_dict,indent=4)).replace("{response}",response)
    answer = []
    count = 0
    while len(answer) != len(sub_questions) and count < 3:
        call_out = openai_chat(Client,"gpt-4o",question=prompt)
        print(call_out)
        # multi_chocie
        if call_out:
            try:
                json_part = call_out.split("Answer Extraction:")[-1].strip().strip("`")
                if json_part.startswith("json"):
                    json_part = json_part[len("json"):].strip()
                extracted_data = json.loads(json_part)
                answer = [sub_data["extracted_answer"] for sub_data in extracted_data]                         
            except  Exception as e:
                print(e)
        count += 1
    return answer


def evaluate_sub_question(main_question:str,
                        sub_questions:list,
                        sub_answers:list,
                        gt_ans:str):
    
    sub_ques_gt_pred= [{"sub_question":sub_q, 
                        "groundtruth_answer":gt_a, 
                        "AI prediction":sub_a, 
                        } for sub_q,sub_a, gt_a in zip(sub_questions,sub_answers,gt_ans) ]
    prompt  = PROMPT_EVAL_BLANK.replace("{main_question}",main_question).replace("{sub_ques_gt_pred}",json.dumps(sub_ques_gt_pred,indent=4))
    score = [] 
    count = 0
    while len(score) != len(gt_ans) and count < 3:
        call_out = openai_chat(Client,"gpt-4o",question=prompt)
        print(call_out)
        
        if call_out:
            try:
                json_part = call_out.split("Score:")[-1].strip().strip("`")
                if json_part.startswith("json"):
                    json_part = json_part[len("json"):].strip()
                score_dict = json.loads(json_part)
                score = [sub_data["score"] for sub_data in score_dict]

            except  Exception as e:
                print(e)
        
        count +=1
    return score

def test_exam_perform_mp(model_out:dict,
                      anno_data:dict,
                      ):
  
    def process_line(anno_item, model_out_line):
        
        if anno_item["type"] == "multiple-choice":
            extracted_answer = model_out_line.get('extracted',[])
            if not extracted_answer:
                extracted_answer = extract_multi_choice_answer(question=anno_item['conversations'][0]['value'],
                                                            response=model_out_line['text'])
            model_out_line['extracted'] = extracted_answer
            gt_ans = [op for ops in anno_item["answer"] for op in ops.lower().strip()]

            if all(item.lower() in gt_ans for item in extracted_answer):
                model_out_line['score'] = [1]
            else:
                model_out_line['score'] = [0]
                    
        elif anno_item["type"] == "fill-in-the-blank":
            pass
            gt_ans =  anno_item['answer']
            extracted_answer = model_out_line.get('extracted',[])
            
            if len(model_out_line.get('score',[])) ==0:
                if not extracted_answer or len(extracted_answer) != len(gt_ans):
                    extracted_answer = extract_multi_blank_answer(
                                                                main_question=anno_item['main_question'],
                                                                sub_questions=anno_item['sub_questions'],
                                                                response=model_out_line['text'])
                model_out_line['extracted'] = extracted_answer
                blank_results = evaluate_sub_question(main_question=anno_item['main_question'],
                                                    sub_questions=anno_item['sub_questions'],
                                                    sub_answers=extracted_answer, 
                                                    gt_ans=gt_ans)
                model_out_line['score'] = blank_results
        
        else:
            raise ValueError     
        return model_out_line
    

    with ThreadPoolExecutor(max_workers=8) as executor: 
        futures = [executor.submit(process_line, anno_item, model_out[question_id]) for question_id, anno_item in anno_data.items() if question_id in model_out]
        for future in futures:
            processed_line = future.result()
            question_id = processed_line['id']
            model_out[question_id] = processed_line
    
    return model_out






def test_exam_perform(model_out:dict,
                      anno_data:dict):

    for question_id, anno_item in tqdm(anno_data.items()):
        if question_id not in model_out:
            print(f"{question_id} not in model_out_file!")
            continue

        if anno_item["type"] == "multiple-choice":
            extracted_answer = model_out[question_id].get('extracted',[])
            if not extracted_answer:
                extracted_answer = extract_multi_choice_answer(question=anno_item['conversations'][0]['value'],
                                                            response=model_out[question_id]['text'])
            model_out[question_id]['extracted'] = extracted_answer
            gt_ans = [op for ops in anno_item["answer"] for op in ops.lower().strip()]

            if all(item.lower() in gt_ans for item in extracted_answer):
                model_out[question_id]['score'] = [1]
            else:
                model_out[question_id]['score'] = [0]

        elif anno_item["type"] == "fill-in-the-blank":
            pass
            gt_ans =  anno_item['answer']
            extracted_answer = model_out[question_id].get('extracted',[])
            # blank_results = model_out[question_id].get('score',[])
            if not extracted_answer or len(extracted_answer) != len(gt_ans):
                extracted_answer = extract_multi_blank_answer(
                                                            main_question=anno_item['main_question'],
                                                            sub_questions=anno_item['sub_questions'],
                                                            response=model_out[question_id]['text'])
            model_out[question_id]['extracted'] = extracted_answer
            blank_results = evaluate_sub_question(main_question=anno_item['main_question'],
                                                  sub_questions=anno_item['sub_questions'],
                                                sub_answers=extracted_answer, 
                                                gt_ans=gt_ans)
            model_out[question_id]['score'] = blank_results    
        else:
            raise ValueError
    return model_out

def evaluate_score(model_out):
    multi_choice_record={
        "total_num":0,
        "correct_num":0,
        "score":0,
        }
    fill_in_the_blank={
        "total_num":0,
        "correct_num":0,
        "score":0,
        "blank_num":0,
        "blank_correct_num":0,
        "blank_score":0           
    } 

    for question_id, anno_item in tqdm(anno_data.items()):


        if anno_item["type"] == "multiple-choice":
            multi_choice_record['total_num'] += 1
            if question_id not in model_out:
                print(f"{question_id} not in model_out_file!")
                continue
            extracted_answer = model_out[question_id].get('extracted',[])
            
            gt_ans = [op for ops in anno_item["answer"] for op in ops.lower().strip()]

            if all(item.lower() in gt_ans for item in extracted_answer):
                multi_choice_record['correct_num'] += 1
                
        elif anno_item["type"] == "fill-in-the-blank":

            fill_in_the_blank['total_num'] += 1
            if question_id not in model_out:
                print(f"{question_id} not in model_out_file!")
                continue
            gt_ans =  anno_item['answer']
            fill_in_the_blank['blank_num'] += len(gt_ans)
            extracted_answer = model_out[question_id].get('extracted',[])
            blank_results = model_out[question_id].get('score',[])
            if all(blank_results):
                fill_in_the_blank['correct_num'] += 1
            
            fill_in_the_blank['blank_correct_num'] += sum(blank_results)
            
        else:
            raise ValueError        


    multi_choice_record['score'] = multi_choice_record['correct_num']/multi_choice_record['total_num']
    fill_in_the_blank['score'] = fill_in_the_blank['correct_num']/fill_in_the_blank['total_num']
    fill_in_the_blank['blank_score'] = fill_in_the_blank['blank_correct_num'] / fill_in_the_blank['blank_num']

    total_num = multi_choice_record['total_num'] + fill_in_the_blank['total_num']
    total_correct = multi_choice_record['correct_num'] + fill_in_the_blank['correct_num']
    total_score = total_correct / total_num

    print(f"multi_choice_record:\n{json.dumps(multi_choice_record, indent=4, ensure_ascii=False)}")
    print(f"fill_in_the_blank:\n{json.dumps(fill_in_the_blank, indent=4, ensure_ascii=False)}")
    print(f"{total_num=}, {total_correct=}, {total_score=:.4f}")

    return {"multi_choice_record":multi_choice_record,
            "fill_in_the_blank":fill_in_the_blank,
            "total":{
                "total_num":total_num,
                "total_correct":total_correct,
                "total_score":total_score
            }}

      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-outfile', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--datasets', type=str, default='smiles_ocr')

    args = parser.parse_args()
    task = args.datasets
    if not args.model_outfile:
        model_outfile = [(filename, datetime.strptime(filename.split(".")[0].split("_")[-1], "%y%m%d%H%M%S")) 
                        for filename in os.listdir(args.out_dir) 
                        if (filename.startswith(args.datasets)) ]
        

        if len(model_outfile) == 0:
            print("Error: No output model file found. Exiting.")
            exit()
        model_outfile, _ = max(model_outfile, key=lambda x: x[1])
        model_outfile = os.path.join(args.out_dir,model_outfile)
    else:
        model_outfile = args.model_outfile

    print(f"Evaluating {model_outfile}...")

    with open(model_outfile,"r",encoding='utf-8') as f:
        model_out = [json.loads(line) for line in f.readlines()]
    
    with open(ds_collections[task]['question'],"r",encoding='utf-8') as f:
        anno_data = [json.loads(line) for line in f.readlines()]
    
    model_out = list2id_dict(model_out)
    anno_data = list2id_dict(anno_data)

    extracted_model_out = test_exam_perform_mp(model_out = model_out, anno_data=anno_data)

    
    with open(model_outfile,"w") as f:
        for line in extracted_model_out.values():
            f.write(json.dumps(line,ensure_ascii=False)+"\n")

    results = evaluate_score(model_out=extracted_model_out)

    score_file = os.path.join(args.out_dir,"score_" + os.path.basename(model_outfile))
    with open(score_file,"w") as f:
        json.dump(results, f, ensure_ascii=False,indent=4)
    
    # with open(model_outfile,"w") as f:
    #     for line in extracted_model_out.values():
    #         f.write(json.dumps(line,ensure_ascii=False)+"\n")