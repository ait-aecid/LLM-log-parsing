import csv
import itertools
from datetime import datetime
import os
import concurrent.futures

import pandas as pd
from Levenshtein import distance

from evaluate.evaluator import evaluate
from evaluate.evaluator_PA import evaluate_PA

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from functions.benchmark_settings import *
from functions.gram import *
from functions.llm_func import *
# from functions.tree import *
from CONSTANT import input_dir,similarity_threshold
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet
import nltk
nltk.download("wordnet")

from PSQL.conConfig import model

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
    elif num_elements == 1:
        random_elements = random.sample(my_logs, num_elements)
        return random_elements
    else:
        distances = []
        for item1, item2 in itertools.combinations(my_logs, 2):
            dist = distance(item1, item2)
            distances.append(((item1, item2), dist))

        distances.sort(key=lambda x: x[1], reverse=True)
        num_elements = int(num_elements / 2)
        top_elements = distances[:num_elements]
        random_elements = []
        for pair, dist in top_elements:
            random_elements.append(pair[0])
        return random_elements

def template_cluster(group_gram, log_content, log_template, file_name):
    logs = []
    prompt_template = './prompt'
    with open(prompt_template, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    group_tempalte = {}
    pattern = re.compile(r'\d*?<\*>[0-9.-]*<\*>\d*?')
    group_logs = []
    for key in group_gram.keys():
        log = get_random_elements(group_gram[key], log_content)
        group_logs.append(log)

    prompts = extract_examples(group_logs, prompt_template, log_template, model)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(call_openai_api, prompts))
    group_gram_keys = list(group_gram.keys())
    input_tokens = 0
    output_tokens = 0
    for index in range(len(group_gram_keys)):
        template = results[index][0]
        input_tokens += results[index][1]
        output_tokens += results[index][2]
        logs.append(template)
        if template in group_tempalte.keys():
            group_tempalte[template] += group_gram[group_gram_keys[index]]
        else:
            group_tempalte[template] = group_gram[group_gram_keys[index]]

    return group_tempalte,logs,input_tokens,output_tokens


if __name__ == '__main__':
    benchmark_result = []
    input_tokens_list = []
    output_tokens_list = []
    for dataset, setting in benchmark_settings.items():
        starttime = datetime.now()
        if True:
            print('\n=== Evaluation on %s ===' % dataset)
            indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
            log_file = os.path.basename(setting['log_file'])

            filepath = os.path.join(indir, log_file)
            print('Parsing file: ' + filepath)
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
            # print(len(group.keys()))
            match, matchedkeys = words_similarity(sorted(group.keys()), similarity_threshold, 0.32)
            mergeRes = pd.DataFrame({"element1": match.keys(), "element2": match.values()})
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
            threshold = 1 / len(group.keys()) * 5
            group = cluster_3_gram(group, threshold)
            log_template = {}
            group, templates, input_tokens, output_tokens = template_cluster(group, log_content, log_template, os.path.dirname(setting['log_file']))
            pLine = 1
            listEID = [0] * 2000
            listtemplate = [0] * 2000
            templateID = 1
            EIDdword = {}
            for key in group.keys():
                for lineID in group[key]:
                    listEID[lineID] = "E" + str(templateID)
                    listtemplate[lineID] = key
                EIDdword["E" + str(templateID)] = key
                templateID += 1

            parsed_template_file = os.path.join(indir, os.path.dirname(setting['log_file']) + '_parsed_templates.csv')
            with open(parsed_template_file, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['ID', 'Template'])
                for key, value in EIDdword.items():
                    csv_writer.writerow([key, value])

            df_parsedlog = pd.DataFrame({"EventId": listEID})
            out_file = os.path.basename(setting['out_file'])
            parsedlog = os.path.join(indir, out_file)
            df_parsedlog.to_csv(parsedlog)
            groundtruth = os.path.join(indir, log_file + '_structured.csv')
            F1_measure, accuracy, PTA, RTA = evaluate(groundtruth, parsedlog)
            benchmark_result.append([dataset, accuracy])
            print("dataset :{}, input_tokens:{}, output_tokens:{}".format(dataset, input_tokens, output_tokens))
            input_tokens_list.append(input_tokens)
            output_tokens_list.append(output_tokens)

    print('\n=== Overall evaluation results ===')
    PA_toatal, PTA_toatal, RTA_toatal = evaluate_PA()
    for i in range(len(benchmark_result)):
        benchmark_result[i].append(PA_toatal[i])
        benchmark_result[i].append(PTA_toatal[i])
        benchmark_result[i].append(RTA_toatal[i])
    avg_ga = 0
    avg_pa = 0
    avg_pta = 0
    avg_rta = 0
    for i in range(len(benchmark_result)):
        avg_ga += benchmark_result[i][1]
        avg_pa += benchmark_result[i][2]
        avg_pta += benchmark_result[i][3]
        avg_rta += benchmark_result[i][4]
    avg_ga /= len(benchmark_result)
    avg_pa /= len(benchmark_result)
    avg_pta /= len(benchmark_result)
    avg_rta /= len(benchmark_result)
    pd_result = pd.DataFrame(benchmark_result, columns=['dataset', 'GA', 'PA', 'PTA', 'RTA'])
    print(pd_result)
    print('avarage GA is {}, avarage PA is {}, avarage PTA is {}, avarage RTA is {}'.format(avg_ga, avg_pa, avg_pta,
                                                                                            avg_rta))


