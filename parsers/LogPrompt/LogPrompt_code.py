import pandas as pd
from typing import List
from tqdm import tqdm
import re
import warnings
import requests
import json
import argparse
import time
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--API_KEY',type=str)#Specify your API key here
parser.add_argument('--dataset',type=str)#excel file path, with column 'log' containing rows of raw logs
parser.add_argument('--strategy',type=str)#prompt strategies, choice between [Self,CoT,InContext]
parser.add_argument('--output_file_name',type=str,default="result.xlsx")
parser.add_argument('--example_file',type=str,default='')# example file for the in-context prompt, a excel file with two columns: [log, label]. The label column should be "normal" or "abnormal".
parser.add_argument('--model',type=str,default="gpt-3.5-turbo")
args=parser.parse_args()
OPENAI_API_KEY=args.API_KEY
INPUT_FILE=args.dataset
PROMPT_STRATEGIES=args.strategy
OUTPUT_FILE=args.output_file_name
EXAMPLE_FILE=args.example_file
MODEL = args.model
#API_URL="https://api.openai.com/v1/chat/completions" if "gpt" in MODEL else "https://api.together.xyz/v1/chat/completions"
warnings.simplefilter(action='ignore', category=FutureWarning)

from openai import OpenAI
from together import Together
import os
    
def get_client(model):
    if "gpt" in model:
        return OpenAI()
    elif "deepseek-ai/DeepSeek-R1" in model:
        return Together(api_key=os.environ["TOGETHER_API_KEY"])
    elif "deepseek-reasoner" in model:
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

CLIENT = get_client(MODEL)

def filter_special_chars_for_F1(s):
    special_chars = r'[^\w\s*]'
    filtered_str = re.sub(special_chars, '', s)
    return filtered_str
def filter_special_characters(input_string):
    return re.sub(r'[^\w\s]', '', input_string).replace('true','').replace('false','')

def generate_prompt(prompt_header,logs: List[str],max_len=1000,no_reason=False) -> List[str]:
    prompt_parts_count=[]
    prompt_parts = []
    prompt=prompt_header
    log_count=0
    startStr=""
    for i, log in enumerate(logs):
        if no_reason:
            startStr+="("+str(i+1)+"x\n"
        else:
            startStr+="("+str(i+1)+"x-y\n"
        log_str = f"({i+1}) {log}"
        log_length = len(log_str)
        prompt_length=len(prompt)
        if log_length > max_len:
            print("warning: this log is too long")

        if prompt_length + log_length <= max_len:
            prompt += f" {log_str}"
            prompt_length += log_length + 1
            log_count+=1
            if i<(len(logs)-1) and (prompt_length+len(logs[i+1]))>=max_len:
                prompt_parts.append(prompt.replace("!!FormatControl!!",startStr).replace("!!NumberControl!!",str(log_count)))
                prompt_parts_count.append(log_count)
                log_count=0
                prompt=prompt_header
                startStr=""
                continue
            if i== (len(logs)-1):
                prompt_parts.append(prompt.replace("!!FormatControl!!",startStr).replace("!!NumberControl!!",str(log_count)))
                prompt_parts_count.append(log_count)
        else:
            if prompt!=prompt_header:
                log_count+=1
                prompt+=f" {log_str}"
                prompt_parts.append(prompt.replace("!!FormatControl!!",startStr).replace("!!NumberControl!!",str(log_count)))
                prompt_parts_count.append(log_count)
            else:
                prompt=prompt.replace("!!FormatControl!!",startStr)
                prompt=f"{prompt} ({i+1}) {log}"
                prompt_parts.append(prompt)
                prompt_parts_count.append(1)
            log_count=0
            prompt=prompt_header
            startStr=""
    return prompt_parts,prompt_parts_count

def filter_numbers(text):
    pattern = r'\(\d+\)'
    return re.sub(pattern, '', text)
def reprompt(raw_file_name,j,df_raw_answer,temperature):
    prompt=df_raw_answer.loc[j,"prompt"]
    invoc_start = time.time()
    parsed_log = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
        max_tokens=4096
    ).choices[0].message.content

    if "deepseek-ai" in MODEL: # deepseek from togetherAI returns the thinking process, we need to remove it
        parsed_log = parsed_log.split("</think>")[-1]

    invoc_time = time.time() - invoc_start

    df_raw_answer.loc[j,"answer"]=parsed_log
    df_raw_answer.to_excel(raw_file_name,index=False)
    return parsed_log, invoc_time

def parse_logs(raw_file_name,prompt_parts: List[str],prompt_parts_count) -> float:
    parsed_logs = []
    t1 = time.time()
    for p,prompt in enumerate(tqdm(prompt_parts)):
        temperature = 0
        while True:
            try:
                parsed_log = CLIENT.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=temperature,
                    max_tokens=4096,
                    #timeout=10
                ).choices[0].message.content
                if "deepseek-ai" in MODEL: # deepseek from togetherAI returns the thinking process, we need to remove it
                    parsed_log = parsed_log.split("</think>")[-1]
                break
            except Exception as e:
                print("Error while parsing:", e)
                temperature += 0.4
                continue

        parsed_logs.append(parsed_log)
    invoc_time = time.time() - t1

    pd.DataFrame(data=list(zip(prompt_parts,parsed_logs)),columns=['prompt','answer']).to_excel(raw_file_name)
    return invoc_time

def extract_log_index(prompts):
    log_numbers=[]
    for prompt in prompts:
        log_number=re.findall(r'\((\d{1,4})\)',prompt.split("Organize your answer to be the following format")[1].split('a binary choice between')[0])
        log_numbers.append(sorted(list(set([int(x) for x in log_number]))))
    return log_numbers

def write_to_excel(raw_file_name,df_raw_answer: pd.DataFrame, logs: List[str], prompt_parts_count) -> tuple:
    prompts=df_raw_answer['prompt'].tolist()
    #log_numbers=extract_log_index(prompts) # NOT RELIABLE!!! 
    ### CUSTOM:
    number_counts = prompt_parts_count
    log_numbers=[[i+1 for i in range(sum(number_counts[:j]), sum(number_counts[:j])+number_counts[j])] for j in range(len(number_counts))]
    ###
    parsed_logs=df_raw_answer['answer'].tolist()
    parsed_logs_per_log = []
    for i, parsed_log in enumerate(parsed_logs):
        log_parts = parsed_log
        parsed_logs_per_log.append(log_parts)

    parsed_logs_df = pd.DataFrame()
    index=0
    invoc_time = 0
    for j, parsed_log in tqdm(enumerate(parsed_logs_per_log)):
        reprompts = 0
        while 1:
            temperature=0.5
            try:
                pattern = r"\({0}\).*?\({1}\)"
                xx_list=[]
                log_number=log_numbers[j]
                for i in range(len(log_number)-1):
                    start=log_number[i]
                    end=log_number[i+1]
                    if start!=end-1:
                        print('start:',start,'end:',end)
                        continue
                    match=re.search(pattern.format(start,end),parsed_log.replace('\n',''))
                    if reprompts <= 3: ### CUSTOM
                        xx=match.group().split(f"({start})")[1].split(f"({end})")[0].strip()
                        xx_list.append(xx)
                    else:
                        xx_list.append("") ###

                last_log_number=log_number[-1]
                pattern=r"\({0}\).*".format(last_log_number)
                match=re.search(pattern,parsed_log.replace('\n',''))
                if reprompts <= 3: ### CUSTOM
                    xx=match.group().split(f"({last_log_number})")[1].strip()
                    xx_list.append(xx)
                else:
                    xx_list.append("") ###
                for parsed_log_part in xx_list:
                    if parsed_log_part ==None: #or parsed_log_part=="":
                        print("None in parsed_log_part.")
                        continue
                    pred_raw=filter_numbers(parsed_log_part).strip(' ')
                    pred_label=pred_raw
                    parsed_logs_df = pd.concat([parsed_logs_df, pd.DataFrame({'Content': [logs[index]], 'EventTemplate': [pred_label]})], ignore_index=True)
                    index+=1
                break
            except Exception as e:
                ### CUSTOM
                reprompts += 1
                print(e,"reprompting...")
                temperature+=0.4
                #print(parsed_log)
                parsed_log, invoc_time_inc = reprompt(raw_file_name,j,df_raw_answer,temperature)
                invoc_time += invoc_time_inc

    
    #parsed_logs_df.to_excel('Aligned_'+raw_file_name,index=False)
    parsed_logs_df.to_csv("result.csv", index=False)
    return invoc_time
    
def read_log(path):
    # read file line by line without "\n"
    with open(path, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return pd.DataFrame(lines, columns=["log"])

if __name__ == "__main__":
    # custom
    if INPUT_FILE.endswith('.log'):
        df = read_log(INPUT_FILE)
    elif INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE).rename(columns={'Content':'log'})
    else:
        raise ValueError("Input file format not supported!!!")
    #print(df)
    np.random.seed(123)
    if PROMPT_STRATEGIES == 'CoT':
        df=df.sample(frac=1).reset_index(drop=True)
        answer_desc="a binary choice between normal and abnormal"
        prompt_header="Classify the given log entries into normal an abnormal categories. Do it with these steps: \
        (a) Mark it normal when values (such as memory address, floating number and register value) in a log are invalid. \
        (b) Mark it normal when lack of information. (c) Never consider <*> and missing values as abnormal patterns. \
        (d) Mark it abnormal when and only when the alert is explicitly expressed in textual content (such as keywords like error or interrupt). \
        Concisely explain your reason for each log. Organize your answer to be the following format: !!FormatControl!!, where x is %s and y is the reason. \
        There are !!NumberControl!! logs, the logs begin: "%(answer_desc)
        logs=df['log'].tolist()
        ########### generate prompts ######################
        prompt_parts,prompt_parts_count = generate_prompt(prompt_header,logs,max_len=3000)
        ########### obtain raw answers from GPT ###########
        parse_logs(OUTPUT_FILE,prompt_parts,prompt_parts_count)
        ########### Align each log with its results #######
        df_raw_answer = pd.read_excel(OUTPUT_FILE)
        write_to_excel(OUTPUT_FILE,df_raw_answer,logs)
    if PROMPT_STRATEGIES == 'InContext':
        df_examples=pd.read_excel(EXAMPLE_FILE)
        df=df.sample(frac=1).reset_index(drop=True)
        answer_desc="a binary choice between 0 and 1"
        examples=' '.join(["(%d) Log: %s. Category: %s"%(i+1,df_examples.loc[i,'log'],int(df_examples.loc[i,'label']=='abnormal')) for i in range(len(df_examples))])
        prompt_header="Classify the given log entries into 0 and 1 categories based on semantic similarity to the following labelled example logs: %s.\
        Organize your answer to be the following format: !!FormatControl!!, where x is %s. There are !!NumberControl!! logs, the logs begin: "%(examples,answer_desc)
        logs=df['log'].tolist()
        ########### generate prompts ######################
        prompt_parts,prompt_parts_count = generate_prompt(prompt_header,logs,max_len=3000,no_reason=True)
        ########### obtain raw answers from GPT ###########
        parse_logs(OUTPUT_FILE,prompt_parts,prompt_parts_count)
        ########### Align each log with its results #######
        df_raw_answer = pd.read_excel(OUTPUT_FILE)
        write_to_excel(OUTPUT_FILE,df_raw_answer,logs)        
    if PROMPT_STRATEGIES == "Self":
        #candidate selection
        #df=df[:100]
        prompt_candidates=[]
        with open('prompt_candidates.txt') as f:
            for line in f.readlines():
                prompt_candidates.append(line.strip('\n'))
        for i,prompt_candidate in enumerate(prompt_candidates):
            print('prompt %d'%(i+1))
            answer_desc="a parsed log template"
            prompt_header = "%s Organize your answer to be the following format: !!FormatControl!!, where x is %s. There are !!NumberControl!! logs, the logs begin: "%(prompt_candidate,answer_desc)
            logs=df['log'].tolist()
            ########### generate prompts ######################
            prompt_parts,prompt_parts_count = generate_prompt(prompt_header,logs,max_len=1000,no_reason=True)
            ########### obtain raw answers from GPT ###########
            invoc_time = parse_logs('Candidate_%d_'%(i+1)+OUTPUT_FILE,prompt_parts,prompt_parts_count)
            ########### Align each log with its results #######
            df_raw_answer = pd.read_excel('Candidate_%d_'%(i+1)+OUTPUT_FILE)
            invoc_time += write_to_excel('Candidate_%d_'%(i+1)+OUTPUT_FILE,df_raw_answer,logs, prompt_parts_count)
            with open("logprompt_time.txt", "w") as f:
                f.write(str(invoc_time))
                f.close()
