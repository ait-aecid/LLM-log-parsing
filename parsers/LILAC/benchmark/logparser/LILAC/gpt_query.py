from openai import OpenAI
from together import Together
import os
import re
import time
import string
import json
from .parsing_cache import ParsingCache
from .post_process import correct_single_template

def get_client(model):
    if "gpt" in model:
        return OpenAI()
    elif "deepseek-ai/DeepSeek-R1" in model:
        return Together(api_key=os.environ["TOGETHER_API_KEY"])
    elif "deepseek-reasoner" in model:
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def infer_llm(instruction, exemplars, query, log_message, model='gpt-3.5-turbo', temperature=0.0, max_tokens=4096):
    client = get_client(model)
    
    if "turbo" in model:
        messages = [
            {"role": "system", "content": "You are an expert of log parsing, and now you will help to do log parsing."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "Sure, I can help you with log parsing."}
        ]

        if exemplars:
            for exemplar in exemplars:
                messages.append({"role": "user", "content": exemplar['query']})
                messages.append({"role": "assistant", "content": exemplar['answer']})
        
        messages.append({"role": "user", "content": query})
    else:
        messages = f"{instruction}\n"
        if exemplars:
            messages += "Here are some examples:\n"
            for exemplar in exemplars:
                messages += f"{exemplar['query']}\n{exemplar['answer']}\n"
        messages += f"Please parse the following log message:\n{query}\n"
    
    retry_times = 0
    print("Model: ", model)
    
    while retry_times < 3:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            print("Exception:", e)
            retry_times += 1
    
    print(f"Failed to get response from OpenAI after {retry_times} retries.")
    
    if exemplars is not None and len(exemplars) > 0:
        if exemplars[0]['query'] != 'Log message: `try to connected to host: 172.16.254.1, finished.`' \
        or exemplars[0]['answer'] != 'Log template: `try to connected to host: {ip_address}, finished.`':
            examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
            return infer_llm(instruction, examples, query, log_message, model, temperature, max_tokens)
    
    return f'Log message: `{log_message}`'



def get_response_from_openai_key(query, examples=[], model='gpt-3.5-turbo-0613', temperature=0.0):
    # Prompt-1
    # instruction = "I want you to act like an expert of log parsing. I will give you a log message enclosed in backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template enclosed in backticks."
    instruction = "I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template delimited by backticks."
    if examples is None or len(examples) == 0:
        examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
    question = 'Log message: `{}`'.format(query)
    responses = infer_llm(instruction, examples, question, query,
                          model, temperature, max_tokens=4096)
    return responses


def query_template_from_gpt(log_message, examples=[], model='gpt-3.5-turbo-0613'):
    if len(log_message.split()) == 1:
        return log_message, False, 0
    # print("prompt base: ", prompt_base)
    invoc_start = time.time()
    response = get_response_from_openai_key(log_message, examples, model)
    invoc_end = time.time()
    # print(response)
    lines = response.split('\n')
    log_template = None
    for line in lines:
        if line.find("Log template:") != -1:
            log_template = line
            break
    if log_template is None:
        for line in lines:
            if line.find("`") != -1:
                log_template = line
                break
    if log_template is not None:
        start_index = log_template.find('`') + 1
        end_index = log_template.rfind('`')

        if start_index == 0 or end_index == -1:
            start_index = log_template.find('"') + 1
            end_index = log_template.rfind('"')

        if start_index != 0 and end_index != -1 and start_index < end_index:
            template = log_template[start_index:end_index]
            return template, True, invoc_end - invoc_start

    print("======================================")
    print("ChatGPT response format error: ")
    print(response)
    print("======================================")
    return log_message, False, invoc_end - invoc_start


def post_process_template(template, regs_common):
    pattern = r'\{(\w+)\}'
    template = re.sub(pattern, "<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            return template, True
    print("Get a too general template. Error.")
    return "", False


def query_template_from_gpt_with_check(log_message, regs_common=[], examples=[], model="gpt-3.5-turbo-0613"):
    template, flag, invoc_time = query_template_from_gpt(log_message, examples, model)
    if len(template) == 0 or flag == False:
        print(f"ChatGPT error")
    else:
        tree = ParsingCache()
        template, flag = post_process_template(template, regs_common)
        if flag:
            tree.add_templates(template)
            if tree.match_event(log_message)[0] == "NoMatch":
                print("==========================================================")
                print(log_message)
                print("ChatGPT template wrong: cannot match itself! And the wrong template is : ")
                print(template)
                print("==========================================================")
            else:
                return template, True, invoc_time
    r1, r2 = post_process_template(log_message, regs_common)
    return r1, r2, invoc_time
