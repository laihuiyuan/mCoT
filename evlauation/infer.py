# -*- coding:utf-8 _*-

import re
import json
import argparse
import torch
from vllm import LLM, SamplingParams
import util


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return s
                   

def extract_answer(sentence):
    sentence = str(sentence).replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*|\d+(?:\s+\d+)?', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])

    return pred_answer


def test_answers(pred_str, answer):
    pred = remove_boxed(util.last_boxed_only_string(pred_str))
    if util.is_equiv(pred, answer):
        return True, pred, answer
    else:
        try:
            if isinstance(pred, str):
                pred = extract_answer(pred)
            answer = float(answer.replace(',',''))
            if abs(pred - answer)<0.001:
                return True, pred, answer
            else:
                return False, pred, answer
        except:
            return False, pred, answer


def read_inputs(lang, file_path, prompt_path):
    answers = []
    questions = []
    prompt = "Question:\n{}\nAnswer:\n{}\n"
    with open(prompt_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split(' || ')
            if line[0]==lang:
                prompt = prompt.format('{}', line[1].strip())
    with open(file_path, 'r', encoding="utf-8") as f:
        f=json.load(f)
        for line in f:
            questions.append(prompt.format(line['question'].strip()))
            answers.append(line['answer'])

    return questions, answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--inp_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--stop', type=str, nargs='+', default=[])
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    another_args = {
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'max_model_len': args.max_num_batched_tokens,
    }

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=num_gpus,
        **another_args)
    print('[info] Model loaded')

    # Sampling params
    sampling_params = SamplingParams(
        top_p=args.top_p,
        stop=args.stop,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty
    )

    # Reading data with prompt template
    questions, answers = read_inputs(args.lang, args.inp_path, args.prompt_path)
    print('[info] Data loaded')

    # Generate outputs
    outputs = llm.generate(questions, sampling_params)
    sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
    print('[info] Generation done')

    acc = 0
    miss = 0.001
    flag = True
    # Write outputs
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(sorted_outputs):
            result, pred, gold = test_answers(output.outputs[0].text, answers[i])
            if result:
                acc += 1
                flag = True
            else:
                flag = False
            f.write(json.dumps({'question': output.prompt,
                                'response': output.outputs[0].text,
                                'prediction': pred,
                                'answer': gold,
                                'result': flag}, ensure_ascii=False) + '\n')

    print('[info] %s | Accuracy: %.4f | Correct %d | Total %d' % (
        args.lang, float(acc / len(answers)), acc, len(answers)))
