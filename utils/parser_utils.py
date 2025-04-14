import sys
import os
import time
import shutil
import pandas as pd
import re
import csv
import json
import chromadb
from functools import reduce
import random
from contextlib import redirect_stdout
import io
from utils.settings import settings
from openai import OpenAI
from together import Together

PROJECT_FOLDER = os.getcwd() + "/"
DATA_FOLDER = PROJECT_FOLDER + 'data/'
OUTPUT_FOLDER = PROJECT_FOLDER + 'output/'
OPENAI_KEY = open(PROJECT_FOLDER + 'keys/openai_key.txt', 'r').read()
TOGETHER_API_KEY = open(PROJECT_FOLDER + 'keys/togetherai_key.txt', 'r').read()
DEEPSEEK_API_KEY = open(PROJECT_FOLDER + 'keys/deepseek_key.txt', 'r').read()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

def get_api_key(model):
    if "gpt" in model:
        return os.environ["OPENAI_API_KEY"]
    elif "deepseek-ai/DeepSeek-R1" in model:
        return os.environ["TOGETHER_API_KEY"]
    elif "deepseek-reasoner" in model:
        return os.environ["DEEPSEEK_API_KEY"]
    else:
        return "ollama"
    
def get_client(model):
    if "gpt" in model:
        return OpenAI()
    elif "deepseek-ai/DeepSeek-R1" in model:
        return Together(api_key=os.environ["TOGETHER_API_KEY"])
    elif "deepseek-reasoner" in model:
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def extract_variables(log, template):
    log = re.sub(r'\s+', ' ', log.strip()) # DS
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)
    if matches:
        return matches.groups()
    else:
        return None

def match_log_to_template(log: str, templates):
    for template in templates:
        if extract_variables(log, template):
            return template
    return None

def generate_logformat_regex(logformat):
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

def logfile_to_dataframe(log_file, log_format):
    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                raise e
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def save_logfile_content(infile_path, log_format, outfile_path="tmp_log_content.log"):
    df_structured = logfile_to_dataframe(infile_path, log_format)
    if log_format: # if format is given then save and use the log content as input
        with open(outfile_path, "w") as f:
            for line in df_structured["Content"]:
                f.write(f"{line}\n")

def get_data_paths(path="data/2k/"):
    data_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "structured_corrected.csv" in file:
                data_paths.append(os.path.join(root, file))
    return data_paths

def capture_prints(func, params):
    buffer = io.StringIO()
    with redirect_stdout(buffer):  # Redirect stdout within this block
        func_out = func(**params)
    return func_out, buffer.getvalue()

def get_data_paths(path="data/2k/", identifier="structured_corrected.csv"):
    data_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if identifier in file:
                data_paths.append(os.path.join(root, file))
    return data_paths

def sample_data(template_path, dataset, shot, corrected_LH):
    print("--------------------------------\n", f"Sampling {shot} shots for dataset", dataset, "\n--------------------------------")
    df = pd.read_csv(template_path)
    if corrected_LH:
      df_noncorrected = pd.read_csv(template_path.replace("_corrected", ""))
      df["Content"] = df_noncorrected["Content"] # corrected data contains mistakes in content
    templates = list(set(df["EventTemplate"]))
    random_templates = random.sample(templates, shot)
    # get a random sample for each template
    df_sample = pd.concat([df[df["EventTemplate"] == template].sample(1) for template in random_templates])
    return df_sample
