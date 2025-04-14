import csv
import psycopg2
from sshtunnel import SSHTunnelForwarder
from datetime import datetime
import os
import concurrent.futures
from PSQL.conConfig import *

import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "true"


from functions.benchmark_settings import *
from functions.gram import *
from functions.llm_func import *
from functions.reslove_unblance import reslove_unblance
# from functions.tree import *
from CONSTANT import input_dir,similarity_threshold
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet

from PSQL.conConfig import model
from is_new_log import belongs_exsist_template, csv_to_dict, dict_to_csv

def word_similarity(word1, word2):
    synset1 = wordnet.synsets(word1)
    synset2 = wordnet.synsets(word2)

    if synset1 and synset2:
        similarity = max(s1.path_similarity(s2) for s1 in synset1 for s2 in synset2)
        return similarity
    else:
        return 0.0

def words_similarity(keys, similarity_threshold, word2word_similarity):
    result = {}
    match_index = []
    match_index_total = []
    match_content = []
    keys = list(keys)
    for index1 in range(len(keys)):
        if index1 not in match_index_total:
            for index2 in range(index1 + 1, len(keys)):
                try:
                    similarity = fuzz.ratio(keys[index1], keys[index2])
                except:
                    print("division by zero")
                if similarity >= similarity_threshold and similarity < 100:
                    if index2 not in match_index_total:
                        match_index.append(index2)
                        match_index_total.append(index2)
                        match_content.append(keys[index2])
                        try:
                            result[keys[index1]].append(keys[index2])
                        except:
                            result[keys[index1]] = []
                            result[keys[index1]].append(keys[index2])
            set1 = set(keys[index1].split())
            if keys[index1] in result.keys():
                delRes = []
                for str1 in result[keys[index1]]:
                    flag = True
                    for i in range(min(len(str1.split()), len(keys[index1].split()))):
                        if flag:
                            if str1.split()[i] != keys[index1].split()[i] and flag:
                                if word_similarity(str1.split()[i], keys[index1].split()[i]) <= word2word_similarity and word_similarity(str1.split()[i], keys[index1].split()[i]) != 0.1:
                                    set1 = set1.intersection(set(str1.split()))
                                    flag = False
                                else:
                                    delRes.append(str1)
                                    match_index.pop(match_content.index(str1))
                                    match_content.pop(match_content.index(str1))
                                    flag = False

                for ind in delRes:
                    result[keys[index1]].pop(result[keys[index1]].index(ind))

    number = 0
    for index in sorted(match_index):
        keys.pop(index - number)
        number += 1
    return result, keys

def get_random_elements(my_list, log_content, num_elements=3):
    my_logs = []
    for line in my_list:
        my_logs.append(log_content[line])
    my_logs = list(set(my_logs))
    if num_elements > len(my_logs):
        num_elements = len(my_logs)
        random_elements = random.sample(my_logs, num_elements)
        return random_elements
    else:
        random_elements = random.sample(my_logs, num_elements)
        return random_elements

def template_cluster(group_gram,log_content,log_template):
    new_logs = 0
    logs = {}
    prompt_template = '../prompt'
    with open(prompt_template, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    group_tempalte = {}
    pattern = re.compile(r'\d*?<\*>[0-9.-]*<\*>\d*?')

    group_logs = []
    for key in group_gram.keys():
        log = get_random_elements(group_gram[key], log_content)
        group_logs.append(log)
        new_logs += len(group_gram[key])

    prompts = extract_examples(group_logs, prompt_template, log_template, model)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(call_openai_api, prompts))
    group_gram_keys = [key for key, value in group_gram.items() if len(value) > 0]
    input_tokens = []
    output_tokens = []
    for index in range(len(group_gram_keys)):
        input_tokens.append(results[index][1])
        output_tokens.append(results[index][2])
        template = results[index][0]
        logs[group_gram_keys[index]] = template
        if template in group_tempalte.keys():
            group_tempalte[template] += group_gram[group_gram_keys[index]]
        else:
            group_tempalte[template] = group_gram[group_gram_keys[index]]

    return group_tempalte, logs, input_tokens, output_tokens, len(prompts), new_logs

def self_evolution(logs_templates):
    with SSHTunnelForwarder(**ssh_config) as tunnel:
        conn = psycopg2.connect(
            host='127.0.0.1',
            port=tunnel.local_bind_port,
            **postgresql_config
        )
        cur = conn.cursor()
        for log in logs_templates.keys():
            log_val = log
            template_val = logs_templates[log]
            vector_val = model.encode(log_val, normalize_embeddings=True)

            cur.execute("""INSERT INTO public.log_template(
                                       "ID", log, template, "logVector")
                                      VALUES (nextval('id_seq'), %s, %s, %s)""",
                        (log_val, template_val, list(vector_val)))
            conn.commit()
        cur.close()
        conn.close()


def onlineWork(logs, oracle, group_token_dir):
    group_tokens = csv_to_dict(group_token_dir)

    for dataset, setting in benchmark_settings.items():
        if dataset == oracle:
            print('\n=== Evaluation on %s ===' % dataset)
            with open(os.path.join(input_dir, setting['buffer_file']), 'w', encoding='utf-8') as file:
                for log in logs:
                    file.write(log + '\n')
            indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
            log_file = os.path.basename(setting['buffer_file'])
            filepath = os.path.join(indir, log_file)
            print('Parsing file: ' + filepath)
            starttime = datetime.now()
            group = {}
            lineNum = 0
            headers, regex = generate_logformat_regex(setting['log_format'])
            log_messages = load_logs(filepath, regex, headers)
            log_content = []
            for key, log in tqdm(log_messages.items(), desc='priori knowledge preprocess'):
                log = log["Content"]
                log_content.append(log)
                domain_pattern = r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
                log = re.sub(domain_pattern, "123", log)
                log = re.sub(r'(/[^/\s]+)+', "123", log)
                if log != '':
                    tokens = re.findall(r'[A-Za-z0-9*]+', log)
                    words = ""
                    words_counter = {}
                    for token in tokens:
                        if all(char.isalpha() or char == '*' for char in token):
                            if len(token) < 4:
                                if not is_common_word(token) or len(token) < 2:
                                    continue
                                else:
                                    if token not in words_counter.keys():
                                        words_counter[token] = 1
                                    else:
                                        words_counter[token] += 1
                                    if words_counter[token] < 4:
                                        words += token
                                        words += " "
                            else:
                                if token not in words_counter.keys():
                                    words_counter[token] = 1
                                else:
                                    words_counter[token] += 1
                                if words_counter[token] < 4:
                                    words += token
                                    words += " "
                    if words not in group.keys():
                        group[words] = []
                        group[words].append(lineNum)
                    else:
                        group[words].append(lineNum)
                    lineNum += 1
            print(len(group.keys()))
            match, matchedkeys = words_similarity(sorted(group.keys()), similarity_threshold, 0.32)
            add_words = []
            for key in match.keys():
                if key not in add_words:
                    for word in match[key]:
                        group[key] += group[word]
                        if word in match.keys():
                            for word1 in match[word]:
                                group[key] += group[word1]
                                group.pop(word1)
                            add_words.append(word)
                        group.pop(word)
            pLine = 1
            for key in group.keys():
                print("{}: {}".format(pLine, key))
                pLine += 1
            threshold = 1 / len(group.keys()) * 5
            group = cluster_3_gram(group, threshold)
            log_template = {}
            group, exsists = belongs_exsist_template(group, group_tokens)
            group, templates, input_tokens, output_tokens,free_times,new_logs = template_cluster(group, log_content, log_template)
            pLine = 1
            for key in group.keys():
                print("{}: {}".format(pLine, key))
                pLine += 1
            group, templates = reslove_unblance(group, templates)
            group_tokens.update(templates)
            dict_to_csv(group_tokens, group_token_dir)
            listEID = [0] * len(logs)
            templateID = 1
            EIDdword = {}
            group_logs_evolution = {}
            for key in group.keys():
                for lineID in group[key]:
                    listEID[lineID] = key
                EIDdword["E" + str(templateID)] = key
                templateID += 1
                group_logs_evolution[log_content[group[key][0]]] = key
            for key in exsists.keys():
                for lineID in exsists[key]:
                    listEID[lineID] = key
                EIDdword["E" + str(templateID)] = key
                templateID += 1
            for key in EIDdword.keys():
                print("{}: {}".format(key, EIDdword[key]))
            parsed_template_file = os.path.join(indir, os.path.dirname(setting['log_file']) + '_parsed_templates.csv')
            with open(parsed_template_file, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['EventId', 'Template'])
                for key, value in EIDdword.items():
                    csv_writer.writerow([key, value])
            print(len(EIDdword))
            df_parsedlog = pd.DataFrame({"Template": listEID})
            out_file = os.path.basename(setting['out_file'])
            parsedlog = os.path.join(indir, out_file)
            if os.path.getsize(parsedlog):
                df_parsedlog.to_csv(parsedlog, mode='a', header=False, index=False)
            else:
                df_parsedlog.to_csv(parsedlog, index=False)
            self_evolution(group_logs_evolution) # update prompt database
            return sum(input_tokens), sum(output_tokens), free_times, new_logs


