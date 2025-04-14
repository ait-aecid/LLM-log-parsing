from wordfreq import word_frequency
from CONSTANT import m

def calculate_probability(n_gram_count, log_count):
    return n_gram_count / log_count

def is_common_word(word, lang='en'):
    freq = word_frequency(word, lang)
    return freq > 1e-4

def get_word_weight(tokens, lang='en'):
    tokens_weight = []
    for word in tokens:
        freq = word_frequency(word, lang)
        if word in ["root"]:
            tokens_weight.append(-2)
        elif freq > 1e-3:
            tokens_weight.append(0)
        elif freq > 1e-4:
            tokens_weight.append(4)
        elif freq > 1e-5:
            tokens_weight.append(3)
        elif freq > 1e-6:
            tokens_weight.append(2)
        elif freq > 1e-7:
            tokens_weight.append(1)
        else:
            tokens_weight.append(0)
    return tokens_weight

def max_sum_index(arr, m):
    if m > len(arr):
        return "Invalid input"
    max_sum = sum(arr[:m])
    current_sum = max_sum
    start_index = 0
    for i in range(1, len(arr) - m + 1):
        current_sum = current_sum - arr[i - 1] + arr[i + m - 1]
        if current_sum > max_sum:
            max_sum = current_sum
            start_index = i

    return max_sum, start_index

def get_gram(token,logs):
    count = 0
    for log in logs:
        if token in log:
            count+=1
    return count

def get_index_for_skipgram(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]

def check_probability(probability, threshold):
    return probability >= threshold

def get_skip_gram(token, logs):
    count = 0
    tokens = token.split()
    for log in logs:
        log = log.split()
        indexs = get_index_for_skipgram(log, tokens[0])
        for index in indexs:
            if index + 2 < len(log) and log[(index + 2) % len(log)] == tokens[1]:
                count += 1
    return count

def max_sum_index(arr, m):
    if m > len(arr):
        return "Invalid input"

    max_sum = sum(arr[:m])
    current_sum = max_sum
    start_index = 0

    for i in range(1, len(arr) - m + 1):
        current_sum = current_sum - arr[i - 1] + arr[i + m - 1]
        if current_sum > max_sum:
            max_sum = current_sum
            start_index = i

    return max_sum, start_index

def get_word_weight(tokens, lang='en'):
    tokens_weight = []
    for word in tokens:
        freq = word_frequency(word, lang)
        if word in ["root"]:
            tokens_weight.append(-2)
        elif freq > 1e-3:
            tokens_weight.append(0)
        elif freq > 1e-4:
            tokens_weight.append(4)
        elif freq > 1e-5:
            tokens_weight.append(3)
        elif freq > 1e-6:
            tokens_weight.append(2)
        elif freq > 1e-7:
            tokens_weight.append(1)
        else:
            tokens_weight.append(0)
    return tokens_weight

def cluster_3_gram(group_original, threshold):  # dir{key:[lineID]}
    print("threshold: ", threshold)
    logs = group_original.keys()
    print("logs: ", logs)
    update_logs = [""] * len(logs)

    log_index = 0
    for log in logs:
        tokens = log.split()
        if len(tokens) > m - 1:
            tokens_weight = get_word_weight(tokens)
            max_value, max_index = max_sum_index(tokens_weight, m)
            for i in range(1, m):
                tokens[max_index] = tokens[max_index] + " " + tokens[max_index + i]
            for i in range(1, m):
                tokens.pop(max_index + 1)
            dynamic = [True] * len(tokens)
            dynamic[max_index] = False  # 设第一字符串为常量

            for index in range(max_index + 1, len(tokens)):
                first = (index - 1) % len(tokens)
                second = (index - 2) % len(tokens)

                if dynamic[first]:
                    if dynamic[second]:
                        probability = calculate_probability(get_gram(tokens[index], logs), len(logs))
                        if check_probability(probability, threshold):
                            dynamic[index] = False
                    else:
                        two_gram = tokens[second] + " " + tokens[index]
                        one_gram = tokens[second]
                        probability = calculate_probability(get_skip_gram(two_gram, logs), get_gram(one_gram, logs))
                        if check_probability(probability, threshold):
                            dynamic[index] = False
                elif dynamic[second]:
                    two_gram = tokens[first] + " " + tokens[index]
                    one_gram = tokens[first]
                    probability = calculate_probability(get_gram(two_gram, logs), get_gram(one_gram, logs))
                    if check_probability(probability, threshold):
                        dynamic[index] = False
                else:
                    three_gram = tokens[second] + " " + tokens[first] + " " + tokens[index]
                    two_gram = tokens[second] + " " + tokens[first]
                    probability = calculate_probability(get_gram(three_gram, logs), get_gram(two_gram, logs))
                    if check_probability(probability, threshold):
                        dynamic[index] = False
            for index in range(0, max_index):
                current_index = max_index - index - 1
                first = (current_index + 1) % (max_index + 1)
                second = (current_index + 2) % (max_index + 1)
                if dynamic[first]:
                    if dynamic[second]:
                        probability = calculate_probability(get_gram(tokens[current_index], logs), len(logs))
                        if check_probability(probability, threshold):
                            dynamic[current_index] = False
                    else:
                        two_gram = tokens[current_index] + " " + tokens[second]
                        one_gram = tokens[second]
                        probability = calculate_probability(get_skip_gram(two_gram, logs), get_gram(one_gram, logs))
                        if check_probability(probability, threshold):
                            dynamic[current_index] = False
                elif dynamic[second]:
                    two_gram = tokens[current_index] + " " + tokens[first]
                    one_gram = tokens[first]
                    probability = calculate_probability(get_gram(two_gram, logs), get_gram(one_gram, logs))
                    if check_probability(probability, threshold):
                        dynamic[current_index] = False
                else:
                    three_gram = tokens[current_index] + " " + tokens[first] + " " + tokens[second]
                    two_gram = tokens[first] + " " + tokens[second]
                    probability = calculate_probability(get_gram(three_gram, logs),
                                                        get_gram(two_gram, logs))
                    if check_probability(probability, threshold):
                        dynamic[current_index] = False
            update_log = ""
            for index1 in range(len(dynamic)):
                if not dynamic[index1]:
                    update_log += tokens[index1]
                    update_log += " "
            update_logs[log_index] = update_log
            log_index += 1
        else:
            update_logs[log_index] = log
            log_index += 1

    cluster = {}
    for index in range(len(update_logs)):
        if update_logs[index] not in cluster.keys():
            cluster[update_logs[index]] = []
            cluster[update_logs[index]].append(index)
        else:
            cluster[update_logs[index]].append(index)
    key = list(group_original.keys())
    del_keys = set()
    for indexlist in cluster.values():
        for index in indexlist[1:]:
            group_original[key[indexlist[0]]] += group_original[key[index]]
            group_original[key[index]] = []
            del_keys.add(key[index])
    uodate_group = {}
    for cluster_key in cluster.keys():
        uodate_group[cluster_key] = group_original[key[cluster[cluster_key][0]]]

    return uodate_group