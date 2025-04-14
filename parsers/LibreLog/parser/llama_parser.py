import os
import sys
import argparse
import torch
import csv
import json
import re
import ast
import pandas as pd
import time
import random
import transformers
# import regex_agent
import textdistance
import regex_manager
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
from datetime import datetime

def replace_bracketed_uppercase(text):
    pattern = r'<[A-Z_]+>'
    replaced_text = re.sub(pattern, '<*>', text)
    return replaced_text.strip()

def get_logs_from_group(group_list):
    logs_from_group = []
    for ele in group_list:
        logs_from_group.append(ele["Content"])
    return logs_from_group


def verify_one_regex(log, regex):
    log = log.replace(",", "")
    regex = regex.replace(",", "")
    try:
        if re.search(regex, log):

            return True
        else:
            return False
    except re.error as e:
        return False

def verify_one_regex_to_match_whole_log(log, regex):
    log = log.replace(",", "")
    regex = regex.replace(",", "")
    regex = f'^{regex}$'
    try:
        if re.search(regex, log):

            return True
        else:
            return False
    except re.error as e:
        return False

def check_and_truncate_regex(pattern):
    parts = re.split(r'(\(\.\*\?\))', pattern)
    wildcards_count = parts.count('(.*?)')

    if wildcards_count > 30:
        index_30th = [i for i, part in enumerate(parts) if part == '(.*?)'][29]
        truncated_parts = parts[:index_30th + 1]
        truncated_pattern = ''.join(truncated_parts)
        return truncated_pattern
    else:
        return pattern


class LogParser:
    def __init__(
            self,
            pipeline,
            regex_manager1,
            model="Meta-Llama-3-8B-Instruct",
            regex_sample=5,
            similarity="jaccard",
            do_self_reflection="True"
    ):
        self.total_time = 0.0
        self.invoc_time = 0.0 ### CUSTOM
        self.n_queries = 0
        self.new_event = 0
        self.model = model
        self.regex_sample = regex_sample
        self.pipeline= pipeline
        self.regex_manager1 = regex_manager1
        self.similarity = similarity
        self.do_self_reflection = do_self_reflection
        print("llama_parser is ready.", flush=True)
    
    def cosine_similarity_distance(self,x, y):
        vectorizer = CountVectorizer()
        x_vector = vectorizer.fit_transform([x])
        y_vector = vectorizer.transform([y])
        similarity = cosine_similarity(x_vector, y_vector)[0][0]
        return 1 - similarity
    
    def jaccard_distance(self,x, y):
        return textdistance.jaccard.normalized_distance(x.split(), y.split())


    def min_distance(self,c_set, t_set):
        D = []
        for c_inst in c_set:
            min_candidate_distance = 1e10
            for t_inst in t_set:
                if self.similarity == "cosine":
                    min_candidate_distance = min(min_candidate_distance, self.cosine_similarity_distance(c_inst, t_inst))
                elif self.similarity == "jaccard":
                    min_candidate_distance = min(min_candidate_distance, self.jaccard_distance(c_inst, t_inst))
                else:
                    raise ValueError("Invalid similarity metric.")
            D.append(min_candidate_distance)
        return D
    
    


    def adaptive_random_sampling(self,logs, k,max_logs=200, similarity_flag=False, dic=False):
        if dic:
            logs = get_logs_from_group(logs)
            
        if max_logs is not None and len(logs) > max_logs:
            logs = random.sample(logs, max_logs)

        if len(logs) < k:
            k = len(logs)
        sample_list = []
        T = []
        if self.similarity == "random":
            sample_list=random.sample(logs, k)
        else:
            for r in range(k):
                if len(sample_list) == 0:
                    i = max(range(len(logs)), key=lambda x: len(logs[x].split()))
                    T.append(logs[i])
                    sample_list.append(logs[i])
                    del logs[i]
                else:
                    candidate_distance = self.min_distance(logs, T)
                    if similarity_flag:
                        best_candidate = min(range(len(candidate_distance)), key=lambda x: candidate_distance[x])
                    else:
                        best_candidate = max(range(len(candidate_distance)), key=lambda x: candidate_distance[x])

                    T.append(logs[best_candidate])
                    sample_list.append(logs[best_candidate])
                    logs.remove(logs[best_candidate])

        return [s.replace(",", "") for s in sample_list]

    def generate_prompt_with_log_list(self, log_list,dic=False):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample,dic=dic)
        message = [
            {
                "role": "system",
                "content": """You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with ‘<*>‘ and output ONE static log template that matches all the logs. Print the input logs’ template delimited by backticks""",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "`try to connected to host: <*>, finished.`",
            },
            {"role": "user", "content": f"Log list: {trimmed_list_log}"},
        ]

        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        return full_prompt,trimmed_list_log
    
    def generate_prompt_with_log_list_chatglm(self, log_list,dic=False):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample,dic=dic)
        messages = [
            {
                "role": "system",
                "content": """You will be provided with a list of logs. You must identify and abstract ALL the dynamic variables in logs with ‘<VARIABLE>‘ and output ONLY ONE static log template that matches all the logs in the log list. Print the input logs’ template delimited by backticks.""",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "Log Template: `try to connected to host: <VARIABLE>, finished.`",
            },
            {"role": "user", "content": f'Log list: {trimmed_list_log}'},
        ]

        return messages,trimmed_list_log
    
    def generate_prompt_with_log_list_mistral(self, log_list,dic=False):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample,dic=dic)
        messages = [
            {
                "role": "user",
                "content": """You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with ‘<*>‘ and output ONE static log template that matches all the logs. Print the input logs’ template delimited by backticks""",
            },
            {
                "role": "assistant",
                "content": "OK!",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "`try to connected to host: <*>, finished.`",
            },
            {"role": "user", "content": f"Log list: {trimmed_list_log}"},
        ]
        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return full_prompt,trimmed_list_log
    
    #### custom ####
    def generate_prompt_with_log_list_custom(self, log_list,dic=False):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample,dic=dic)
        messages = [
            {
                "role": "user",
                "content": """You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with ‘<*>‘ and output ONE static log template that matches all the logs. Print the input logs’ template delimited by backticks""",
            },
            {
                "role": "assistant",
                "content": "OK!",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "`try to connected to host: <*>, finished.`",
            },
            {"role": "user", "content": f"Log list: {trimmed_list_log}"},
        ]
        return messages,trimmed_list_log
    ####

    def check_pre_logs(self, log_list,dic=False):
        if dic:
            log_list = get_logs_from_group(log_list)
        log_unique_list = list(set(log_list))
        first_element = log_unique_list[0]
        if len(log_unique_list) != 1:
            return False
        elif (
                (" is " in first_element)
                or ("=" in first_element)
                or (" to " in first_element)
                or ("_" in first_element)
                or ("-" in first_element)
                or (":" in first_element)
                or ("." in first_element)
                or any(char.isdigit() for char in first_element)
        ):
            return False
        return True
    
    def check_long_logs(self, log_list,dic=False):
        if dic:
            log_list = get_logs_from_group(log_list)
        if len(log_list[0].split())>100 and verify_one_regex(log_list[0],"Warning: we failed to resolve data source name (.*?)$"):
            return True
        return False

    def template_to_regex(self, template):
        if "deepseek-ai" in self.model:
            template = template.split("</think>")[-1]
        template = template.strip()
        if "chatglm" in self.model:
            template=replace_bracketed_uppercase(template.replace("Log Template: ","").strip())
        while template.startswith("```") and template.endswith("```"):
            template = template[4:-4]
        while template.startswith("`"):
            template = template[1:]
        while template.endswith("`"):
            template = template[:-1]
        while template.endswith("."):
            template = template[:-1]
        while template.endswith("<*"):
            template = template + ">"
        while template.endswith("<"):
            template = template + "*>"
        while template.endswith("\\"):
            template = template[:-1]
        template = re.sub(r'\<\*\d+\*\>', "<*>", template)
        template = re.sub(r'\<\*\d+\*', "<*>", template)
        template = re.sub(r'\<\*\d+', "<*>", template)
        template = re.sub(r'\<\*\d+\*\>', "<*>", template)
        template = template.replace('*<>', "<*>").replace('*<*>', "<*>").replace('<*>*', "<*>").replace('<>*',
                                                                                                        "<*>").replace(
            '<*|*>', "<*>").replace('<*>>', "<*>").replace('<<*>', "<*>").replace('<*1*>', "<*>").replace('<>',
                                                                                                          "<*>").replace(
            '<*>.', "<*>").replace(",", "")
        template = re.sub(r'(?!<)\*(?!>)', "<*>", template)
        template = re.sub(r'(?<!<)\*>', "<*>", template)
        escaped = re.escape(template)
        regex_pattern = re.sub(r'<\\\*>', r'(.*?)', escaped)
        regex_pattern = re.sub(r'(\(\.\*\?\))+', r'(.*?)', regex_pattern)
        regex_pattern = regex_pattern.replace("\ ", " ")
        regex_pattern = re.sub('(\(\.\*\?\) ){10,}', '(.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?)', regex_pattern, 0)
        regex_pattern = check_and_truncate_regex(regex_pattern)
        return regex_pattern

    def generalize_regex(self, target_string, regex_pattern):
        try:
            option_patterns = re.findall(r'\(\?\:(.*?)\)', regex_pattern)
            for option_pattern in option_patterns:
                options = option_pattern.split('|')
                for option in options:
                    modified_pattern = regex_pattern.replace(f"(?:{option_pattern})", option)
                    if re.match(modified_pattern, target_string):
                        return modified_pattern
        except:
            return regex_pattern
        return regex_pattern

    def correct_single_template(self, template, user_strings=None):
        path_delimiters = {  
            r'\s', r'\,', r'\!', r'\;', r'\:',
            r'\=', r'\|', r'\"', r'\'',
            r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
        }
        token_delimiters = path_delimiters.union({  
            r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
        })
        template = template.replace("proxy\.((?:[^.]+|\.)*(?:\.-?\d+)+):[0-9]+", "(.*?)")
        template = template.replace("\proxy\.([^.]+):(?:-?\d+|443)", "(.*?)")
        template = template.replace("(?:.*?:-?\d+)?", "(.*?)")
        template = template.replace("(.*?|.*)", "(.*?)")
        template = template.replace("(?:\\n|$)", "$")
        template = template.replace("(\\b)", "").replace("\\b", "").replace("(\\n)", "").replace("\\n", "").replace(
            "(?i)",
            "").replace(
            "?i", "").replace("(\\r)", "").replace("\\r", "")

        template = template.strip()
        template = re.sub(r'\s+', ' ', template)

        tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  
        new_tokens = []
        for token in tokens:

            if re.match(r'^\d+$', token):
                token = '(.*?)'

            if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
                if token != '(.*?)/(.*?)':  
                    token = '(.*?)'

            new_tokens.append(token)

        template = ''.join(new_tokens)

        while True:
            prev = template
            template = re.sub(r'<\*>\.<\*>', '(.*?)', template)
            if prev == template:
                break

        while True:
            prev = template
            template = re.sub(r'<\*><\*>', '(.*?)', template)
            if prev == template:
                break

        while template.endswith("\\"):
            template = template[:-1]
        while " #(.*?)# " in template:
            template = template.replace(" #(.*?)# ", " (.*?) ")

        while " #(.*?) " in template:
            template = template.replace(" #(.*?) ", " (.*?) ")

        while "(.*?):(.*?)" in template:
            template = template.replace("(.*?):(.*?)", "(.*?)")

        while "(.*?)#(.*?)" in template:
            template = template.replace("(.*?)#(.*?)", "(.*?)")

        while "(.*?)/(.*?)" in template:
            template = template.replace("(.*?)/(.*?)", "(.*?)")

        while "(.*?)@(.*?)" in template:
            template = template.replace("(.*?)@(.*?)", "(.*?)")

        while "(.*?).(.*?)" in template:
            template = template.replace("(.*?).(.*?)", "(.*?)")

        while ' "(.*?)" ' in template:
            template = template.replace(' "(.*?)" ', ' (.*?) ')

        while " '(.*?)' " in template:
            template = template.replace(" '(.*?)' ", " (.*?) ")

        while "(.*?)(.*?)" in template:
            template = template.replace("(.*?)(.*?)", "(.*?)")
        
        return template

    def replace_nth(self, s, old, new, n):
        parts = s.split(old)
        if len(parts) <= n:
            return s
        return old.join(parts[:n]) + new + old.join(parts[n:])

    def check_and_modify_regex(self, regex, string):
        try:
            pattern = re.compile(regex)
            match = pattern.match(string)
        except:
            return regex

        if not match:
            return regex
        groups = match.groups()
        if len(groups) != 0 and groups[-1] == "":
            regex = regex + "$"
        try:
            pattern = re.compile(regex)
            match = pattern.match(string)
            groups = match.groups()
        except:
            return regex
        modified_regex = regex
        for i, group in enumerate(groups, start=1):
            # group = groups[i]
            if group != None and re.fullmatch(r'\*+', group):
                replacement = '\\' + '\\'.join(list(group))
                modified_regex = self.replace_nth(modified_regex, '(.*?)', replacement, i)
            if group != None and group.endswith(" "):
                replacement = '(.*?) '
                modified_regex = self.replace_nth(modified_regex, '(.*?)', replacement, i)
            if group != None and group.startswith(" "):
                replacement = ' (.*?)'
                modified_regex = self.replace_nth(modified_regex, '(.*?)', replacement, i)
        return modified_regex

    def clean_regex(self, log, regex):
        regex = regex.replace('\d+\.\d+',
                                  '\d+(\.\d+)?') \
            .replace('\\d+', '-?\\d+') \
            .replace('a-f', 'a-z') \
            .replace('A-F', 'A-Z')
        regex = self.correct_single_template(regex)
        if log:
            regex = self.generalize_regex(log, regex)
            regex = self.check_and_modify_regex(regex, log)
        return regex

    def generate_log_template_using_pipeline(self,
                                             log_list,
                                             dic=False,
                                             do_sample=False,
                                             max_new_tokens=1024,
                                             ):
        if self.check_pre_logs(log_list,dic=dic):
            if dic:
                log_list = get_logs_from_group(log_list)
            return re.escape(log_list[0]).replace("\ ", " ")
        try:
            ### custom ###
            messages, sampled_log_list = self.generate_prompt_with_log_list_custom(log_list,dic=dic)
            invoc_start = time.time()
            response = self.pipeline.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096
            ).choices[0].message.content
            if "deepseek-ai" in self.model: # deepseek from togetherAI returns the thinking process, we need to remove it
                response = response.split("</think>")[-1]
            self.invoc_time += time.time() - invoc_start
            self.n_queries += 1
            return self.clean_regex(sampled_log_list[0],self.template_to_regex(response))
            ####
            if "chatglm" in self.model:
                messages, sampled_log_list = self.generate_prompt_with_log_list_chatglm(log_list,dic=dic)
                response, history = self.pipeline[0].chat(self.pipeline[1], messages[-1]["content"], history=messages[:-1],do_sample=do_sample)
                return self.clean_regex(sampled_log_list[0],self.template_to_regex(response))
            elif "Mistral" in self.model or "codegemma" in self.model:
                prompt, sampled_log_list = self.generate_prompt_with_log_list_mistral(log_list,dic=dic)
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                return self.clean_regex(sampled_log_list[0],self.template_to_regex(outputs[0]["generated_text"][len(prompt):]))
            else:
                prompt, sampled_log_list = self.generate_prompt_with_log_list(log_list,dic=dic)
                terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=do_sample,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id,
                )
                out=outputs[0]["generated_text"][len(prompt):]
                resul=self.clean_regex(sampled_log_list[0],self.template_to_regex(out))
                return resul
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory, try to reduce the number of samples",flush=True)
            self.regex_sample=self.regex_sample-1
            ### custom ####
            messages, sampled_log_list = self.generate_prompt_with_log_list_custom(log_list,dic=dic)
            invoc_start = time.time()
            response = self.pipeline.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096
            ).choices[0].message.content
            if "deepseek-ai" in self.model: # deepseek from togetherAI returns the thinking process, we need to remove it
                response = response.split("</think>")[-1]
            self.invoc_time += time.time() - invoc_start
            self.n_queries += 1
            return self.clean_regex(sampled_log_list[0],self.template_to_regex(response))
            ####
            if "chatglm" in self.model:
                messages, sampled_log_list = self.generate_prompt_with_log_list_chatglm(log_list,dic=dic)
                response, history = self.pipeline[0].chat(self.pipeline[1], messages[-1]["content"], history=messages[:-1],do_sample=do_sample)
                self.regex_sample=self.regex_sample+1
                return self.clean_regex(sampled_log_list[0],self.template_to_regex(response))
            elif "Mistral" in self.model or "codegemma" in self.model:
                prompt, sampled_log_list = self.generate_prompt_with_log_list_mistral(log_list,dic=dic)
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                self.regex_sample=self.regex_sample+1
                return self.clean_regex(sampled_log_list[0],self.template_to_regex(outputs[0]["generated_text"][len(prompt):]))
            else:
                prompt, sampled_log_list = self.generate_prompt_with_log_list(log_list,dic=dic)
                terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=do_sample,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id,
                )
                self.regex_sample=self.regex_sample+1
                return self.clean_regex(sampled_log_list[0],self.template_to_regex(outputs[0]["generated_text"][len(prompt):]))
            
            
    def get_logs_from_group(self, group_list):
        logs_from_group = []
        for ele in group_list:
            logs_from_group.append(ele["Content"])
        return logs_from_group

    def find_longest_backtick_content(self, text):
        is_multiline = '\n' in text
        if is_multiline:
            matches = re.findall(r"`(.*?)`", text)
            if not matches or matches == [""]:
                matches = re.findall(r"`(.*)", text)
            if matches:
                return max(matches, key=len)
        return text

    def clean_generated_regex(self,log, template):
        template = self.find_longest_backtick_content(template)
        template = template.strip()
        while template.startswith("`"):
            template = template[1:]
        while template.startswith('"'):
            template = template[1:]
        while template.startswith("^"):
            template = template[1:]
        while template.startswith("\b"):
            template = template[2:]

        while template.endswith("`"):
            template = template[:-1]
        while template.endswith("."):
            template = template[:-1]
        while template.endswith('"'):
            template = template[:-1]
        while template.endswith("$"):
            template = template[:-1]
        while template.endswith("\b"):
            template = template[:-2]
        while template.endswith("finished"):
            template = template[:-8]
        template = template.replace(",", "")
        template = template.replace("\ ", " ")
        return self.clean_regex(log =log, regex=template)

    def check_regex_from_groups(self, res_list, groups_dict_list, log_regex, new_event=0):
        wrong_logs = []
        for log in groups_dict_list:
            if self.do_self_reflection == "True":
                if verify_one_regex(log["Content"], log_regex):
                    self.regex_manager1.add_regex_template(log_regex,log["Content"])
                    if new_event == 0:                    
                        res_list.append((log["Content"], log["EventId"], log_regex))
                    else:
                        res_list.append((log["Content"], new_event, log_regex))
                else:
                    wrong_logs.append(log)
            else:
                if verify_one_regex(log["Content"], log_regex):
                    self.regex_manager1.add_regex_template(log_regex,log["Content"])
                res_list.append((log["Content"], log["EventId"], log_regex))
        return res_list, wrong_logs

    def store_regx_for_logs(self, res_list, groups_dict_list, log_regex):
        res_list, wrong_logs = self.check_regex_from_groups(
            res_list, groups_dict_list, log_regex
        )
        len_wrong = len(wrong_logs)
        test_time = 0
        while len(wrong_logs) > 0 and (test_time < 3 and len_wrong == len(wrong_logs)):
            len_wrong = len(wrong_logs)
            test_time = test_time + 1
            log_regex = self.generate_log_template_using_pipeline(
                log_list=wrong_logs,dic=True
            )
            res_list, wrong_logs = self.check_regex_from_groups(
                res_list, wrong_logs, log_regex, self.new_event
            )
            if len_wrong != len(wrong_logs):
                self.new_event = self.new_event + 1
        for log in wrong_logs:
            res_list.append((log["Content"], self.new_event, log_regex))
            self.new_event = self.new_event + 1
        return res_list
    
    def remove_first_matching_item(self,data, content_to_remove):
        for i, item in enumerate(data):
            if item['Content'] == content_to_remove:
                del data[i]
                break  
        return data

    def parse(self, groups_from_parser, logs):
        res_list = []
        start_time = datetime.now()
        if self.check_pre_logs(log_list=logs, dic=False):
            res_list = self.store_regx_for_logs(res_list, groups_from_parser, re.escape(logs[0]).replace("\ ", " "))
        else:
            for log in logs[::-1]:
                matched_regex = self.regex_manager1.find_matched_regex_template(log)
                if matched_regex:
                    logs.remove(log)
                    groups_from_parser = self.remove_first_matching_item(groups_from_parser, log)
                    res_list.append([log, "0", matched_regex])
            if logs:
                log_regex = self.generate_log_template_using_pipeline(log_list=logs)
                res_list = self.store_regx_for_logs(res_list, groups_from_parser, log_regex)
        time_taken = datetime.now() - start_time
        self.total_time += time_taken.total_seconds()
        return res_list

    def print_time(self):
        print("[LLaMa parsing time taken: {!s}]".format(self.total_time), flush=True)
