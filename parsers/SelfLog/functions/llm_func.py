import json
import random
import re
import time
import os

import numpy as np
import openai
from openai import OpenAI
from together import Together
from tqdm import tqdm

from llmAPIsetting import *
from PSQL.findTopKexam import findTopK
from scipy.spatial.distance import cosine

def filter_sensitive_information_in_list(text_list):

    ip_port_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b')
    linux_path_pattern = re.compile(r'/\S+')
    windows_path_pattern = re.compile(r'[A-Za-z]:\\(?:\S+\\)*\S*')
    time_pattern = re.compile(r'\b\d{4}(?:-\d{2}(?:-\d{2}(?: \d{2}:\d{2}(?::\d{2})?)?)?)?\b')
    win_path_pattern = re.compile(r"from path '\\?\\.*'")
    filtered_text_list = []
    for text in text_list:
        text_with_time_replaced = time_pattern.sub('<*>', text)
        filtered_text = ip_port_pattern.sub('<*>:<*>', text_with_time_replaced)
        filtered_text = linux_path_pattern.sub('<*>', filtered_text)
        filtered_text = windows_path_pattern.sub('<*>', filtered_text)
        filtered_text = win_path_pattern.sub('from path <*>', filtered_text)
        filtered_text_list.append(filtered_text)

    return filtered_text_list

def extract_examples(my_list,prompt_template,log_template,model):
    prompts = []

    for list in tqdm(my_list):
        if len(list) == 0:
            continue
        # examples = findTopK(list[0], model)
        examples = []
        grouplogs = ""
        for example in examples:
            grouplogs += "Example: log: " + example["log"] + "\n"
            grouplogs += "logTemplate: " + example["template"] + "\n"
        log_list = list
        logs = ""
        for index in range(len(log_list)):
            logs += str(index) + ": "+ log_list[index] + "\n"
        # prompt = prompt_template.format(groupLogs=grouplogs, inputlogs=logs) # similarLogEntry=existlog, similarLogTemplate=exsistTemplate,
        prompt = prompt_template.format(inputlogs=logs)
        prompts.append(prompt)
    return prompts  #, keys

def most_similar_template(input_log,log_template,model):
    if not log_template:
        return "", ""
    keys = list(log_template.keys())
    keys_embdding = []
    for key in keys:
        keys_embdding.append(log_template[key][0])
    vector = model.encode(input_log, normalize_embeddings=True)
    most_similar_index = -1
    highest_similarity = -1

    for i, vec in enumerate(keys_embdding):
        if np.all(vec == 0) or np.all(vector == 0):
            continue

        similarity = 1 - cosine(vector, vec)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_index = i
    return keys[most_similar_index], log_template[keys[most_similar_index]][1]

#### custom ####
def get_client(model):
    if "gpt" in model:
        return OpenAI()
    elif "deepseek-ai/DeepSeek-R1" in model:
        return Together(api_key=os.environ["TOGETHER_API_KEY"])
    elif "deepseek-reasoner" in model:
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
####

def call_openai_api(prompt, llm_model):
    input_tokens = 0
    output_tokens = 0
    max_retries = 3

    answer = ""
    client = get_client(llm_model)
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4096
            )
            answer = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            break
        except openai.OpenAIError as e:
            if '504' in str(e):
                print(f"Retry {retry + 1} after waiting...")
                time.sleep(5)
            else:
                print(prompt)
                raise
    try:
        answer = json.loads(answer)
    except:
        pattern = r'"?logTemplate"?: "(.+)"'
        match = re.search(pattern, answer)
        if match:
            answer = {}
            answer["logTemplate"] = match.group(1)
        else:
            answer = {}
            answer["logTemplate"] = "LLM output wrong"
    pattern = re.compile(r'Input\(logs\):(.*?)(?:\n|$)', re.DOTALL)
    match = pattern.search(prompt)
    extracted_content = "empty"
    if match:
        extracted_content = match.group(1).strip()
    try:
        if answer["logTemplate"] == "LLM output wrong" or answer["logTemplate"] is None:
            answer["logTemplate"] = extracted_content
        #print(extracted_content, answer["logTemplate"])
    except:
        pass
        #print(answer)
    return answer["logTemplate"], input_tokens, output_tokens

