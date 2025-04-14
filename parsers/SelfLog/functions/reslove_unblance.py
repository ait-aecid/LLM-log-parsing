import re

def find_common_prefix(string_list):
    result = {}

    for i in range(len(string_list)):
        current_string = re.split(' |=', string_list[i]) #string_list[i].split()
        common_prefixes = []

        for j in range(len(string_list)):
            if i != j:
                other_string = re.split(' |=',string_list[j]) #string_list[j].split()
                common_prefix = []

                mask_num = 0
                for k in range(min(len(current_string), len(other_string))):
                    if current_string[k] == other_string[k]:
                        common_prefix.append(current_string[k])
                    elif current_string[k] == "<*>" or other_string[k] == "<*>":
                        common_prefix.append(current_string[k])
                        mask_num += 1
                    else:
                        break
                if mask_num/len(current_string) > 0.7:
                    common_prefix = []

                if common_prefix:
                    common_prefixes.append({"prefix": " ".join(common_prefix), "index": j})

        result[string_list[i]] = common_prefixes

    return result

def fix_sorting(templates):
    processed = []
    common_prefix = find_common_prefix(templates)
    num = 0
    for json_pre_ind in common_prefix.values():
        longkey = 0
        longindex = [num]
        for dic in json_pre_ind:
            if len(dic["prefix"].split()) > longkey:
                longindex = [num]
                longkey = len(dic["prefix"].split())
                longindex.append(dic["index"])
            elif len(dic["prefix"].split()) == longkey:
                longindex.append(dic["index"])
        num += 1
        processed.append(sorted(longindex))
    value_to_index = {}
    for i, sublist in enumerate(processed):
        values = " ".join(map(str, sublist))
        if values not in value_to_index:
            value_to_index_keys = list(value_to_index.keys())
            make_new_key = True
            for key in value_to_index_keys:
                if set(values.split()).issubset(set(key.split())):
                    values = key
                    make_new_key = False
                    break
                elif set(key.split()).issubset(values.split()):
                    value_to_index[values] = value_to_index[key]
                    value_to_index.pop(key)
                    make_new_key = False
                    break
            if make_new_key:
                value_to_index[values] = []
        value_to_index[values].append(i)

    result = []
    for sublist in value_to_index.values():
        lengthsplit = {}
        for index in sublist:
            lengthsplit[index] = len(templates[index].split())
        lengthsplit = {k: v for k, v in sorted(lengthsplit.items(), key=lambda item: item[1],reverse=True)}
        for key in lengthsplit.keys():
            result.append(templates[key])

    return result

def reslove_unblance(group_unblance, group_tokens):
    delimiters = [' ', '=']
    pattern = '|'.join(map(re.escape, delimiters))
    keys = list(group_unblance.keys())
    templates = sorted(keys)
    templates = fix_sorting(templates)
    templates_index = []
    templates_groups = {}
    for tem in templates:
        templates_index.append(keys.index(tem))
    logs_with_marked_equal = [re.sub('=', ' <EQUAL> ', log) for log in templates]
    logs_with_marked_equal = [re.sub(r'\s+', ' ', log) for log in logs_with_marked_equal]
    logs_with_marked_equal = [re.sub(r'"<*>"', '<*>', log) for log in logs_with_marked_equal]
    logs_with_marked_equal = [re.sub(r":(?=\S)", ': ', log) for log in logs_with_marked_equal]
    all_words = [re.split(pattern, log) for log in logs_with_marked_equal]
    for index in range(len(all_words)):
        if index == len(all_words) -1:
            break
        tokens1 = all_words[index]
        tokens2 = all_words[index+1]
        flagFadeBack = False
        if len(tokens1) == len(tokens2):
            mask_index = []
            unmask_num = 0
            same_num = 0
            for index1 in range(len(tokens1)):
                if tokens1[index1] == tokens2[index1]:
                    same_num += 1
                    continue
                elif all(char == '*' for char in tokens1[index1]) or all(char == '*' for char in tokens2[index1]):
                    same_num += 1
                    continue
                elif tokens1[index1] == "<*>" or tokens2[index1] == "<*>" or tokens1[index1] == "<*>," or tokens2[index1] == "<*>,":
                    mask_index.append(index1)
                    if tokens1[index1] == "<*>":
                        flagFadeBack = True
                        frontIndex = index
                else:
                    unmask_num += 1
            if unmask_num < 3 and same_num > 0:
                for mask in mask_index:
                    all_words[index][mask] = "<*>"
                    all_words[index+1][mask] = "<*>"
            while flagFadeBack:
                frontIndex -= 1
                tokens1 = all_words[frontIndex]
                tokens2 = all_words[frontIndex + 1]
                if len(tokens1) == len(tokens2):
                    mask_index = []
                    unmask_num = 0
                    same_num = 0
                    for index1 in range(len(tokens1)):
                        if tokens1[index1] == tokens2[index1]:
                            same_num += 1
                            continue
                        elif all(char == '*' for char in tokens1[index1]) or all(
                                char == '*' for char in tokens2[index1]):
                            same_num += 1
                            continue
                        elif tokens1[index1] == "<*>" or tokens2[index1] == "<*>":
                            mask_index.append(index1)
                            if tokens1[index1] == "<*>":
                                flagFadeBack = True
                                frontIndex = index
                        else:
                            unmask_num += 1
                    if unmask_num < 3 and same_num > 0:
                        for mask in mask_index:
                            all_words[frontIndex][mask] = "<*>"
                            all_words[frontIndex + 1][mask] = "<*>"
                    else:
                        flagFadeBack = False
                else:
                    flagFadeBack = False

    templates = [' '.join(log).replace(' <EQUAL> ', '= ') for log in all_words]
    for index in range(len(templates)):
        if templates[index] not in templates_groups.keys():
            templates_groups[templates[index]] = group_unblance[keys[templates_index[index]]]
        else:
            templates_groups[templates[index]] += group_unblance[keys[templates_index[index]]]

    for index in templates_index:
        try:
            group_tokens[list(group_tokens.keys())[index]] = templates[index]
        except:
            break
    templates_groups = fix_strict_match(templates_groups)
    return templates_groups, group_tokens

def fix_strict_match(templates_groups):
    keys = list(templates_groups.keys())
    key_tokens = {}
    for key in keys:
        keyDell = key.replace("<*>", "")
        token_str = ""
        tokens = re.findall(r'[A-Za-z0-9*=]+', keyDell)
        for token in tokens:
            if not all(char.isnumeric() for char in token):
                token_str += token
        if token_str not in key_tokens.keys():
            key_tokens[token_str] = [keys.index(key)]
        else:
            key_tokens[token_str].append(keys.index(key))
    result = {}
    for key in key_tokens.keys():
        if len(key_tokens[key]) == 1:
            result[keys[key_tokens[key][0]]] = templates_groups[keys[key_tokens[key][0]]]
        else:
            result[keys[key_tokens[key][1]]] = []
            for index in key_tokens[key]:
                result[keys[key_tokens[key][1]]] += templates_groups[keys[index]]
    return result