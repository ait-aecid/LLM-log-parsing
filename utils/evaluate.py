from nltk.metrics.distance import edit_distance
from collections import Counter
import pandas as pd
import os

# # calculate PA
# def evaluatePA(groundtruth, result):
#     # len(predicted_list) may smaller than len(groundtruth)
#     length = len(result["EventTemplate"])
#     if length == 0: return 0
#     correct = 0
#     for i in range(length):
#         if groundtruth['Content'][i] != result["Content"][i]:
#             raise ValueError(f"Log content does not match at index {i}! Mismatched content:\n{groundtruth['Content'][i]}\n{result['Content'][i]}")
#         if result["EventTemplate"][i] == groundtruth['EventTemplate'][i]:
#             correct += 1
#     return correct/length

# calculate parsing accuracy
def evaluatePA(groundtruth, result):
    # len(predicted_list) may smaller than len(groundtruth)
    length = len(result['EventTemplate'])
    if length == 0: return 0
    correct = 0
    try:
        for i in range(length):
            if result['EventTemplate'][i] == groundtruth.loc[groundtruth['Content'] == result['Content'][i]]['EventTemplate'].values[0]:
                correct += 1
    except:
        raise ValueError(f"{result["Content"][i]} not found in groundtruth!")
    return correct/length

# calculate parsing accuracy
def evaluatePA_no_whitespace(groundtruth, result):
    # len(predicted_list) may smaller than len(groundtruth)
    length = len(result['EventTemplate'])
    if length == 0: return 0
    correct = 0
    try:
        for i in range(length):
            if result['EventTemplate'][i].replace(" ", "") == groundtruth.loc[groundtruth['Content'] == result['Content'][i]]['EventTemplate'].values[0].replace(" ", ""):
                correct += 1
    except:
        raise ValueError(f"{result["Content"][i]} not found in groundtruth!")
    return correct/length

# calculate GA
def evaluateGA(groundtruth, result):
    # load logs and templates
    compared_list = result["Content"].tolist()

    # select groundtruth logs that have been parsed
    parsed_idx = []
    for idx, row in groundtruth.iterrows():
        if row['Content'] in compared_list:
            parsed_idx.append(idx)
            compared_list.remove(row['Content'])

    if not (len(parsed_idx) == 2000):
        print(len(parsed_idx))
        print("Wrong number of groundtruth logs!")
        return 0

    groundtruth = groundtruth.loc[parsed_idx]

    # grouping
    groundtruth_dict = {}
    for idx, row in groundtruth.iterrows():
        if row['EventTemplate'] not in groundtruth_dict:
            # create a new key
            groundtruth_dict[row['EventTemplate']] = [row['Content']]
        else: 
            # add the log in an existing group
            groundtruth_dict[row['EventTemplate']].append(row['Content'])

    result_dict = {}
    for idx, row in result.iterrows():
        if row["EventTemplate"] not in result_dict:
            # create a new key
            result_dict[row["EventTemplate"]] = [row["Content"]]
        else: 
            # add the log in an existing group
            result_dict[row["EventTemplate"]].append(row["Content"])

    # sorting for comparison
    for key in groundtruth_dict.keys():
        groundtruth_dict[key].sort()

    for key in result_dict.keys():
        result_dict[key].sort()

    # calculate GA
    count = 0
    for parsed_group_list in result_dict.values():
        for gt_group_list in groundtruth_dict.values():
            if parsed_group_list == gt_group_list:
                count += len(parsed_group_list)
                break

    return count / 2000

def evaluate_template_level(df_groundtruth, df_parsedresult, filter_templates=None, no_whitespace=False):
    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]
    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedresult['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('parsedlog')
    
    for identified_template, group in grouped_df:
        corr_oracle_templates = set(list(group['groundtruth']))
        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)

        if no_whitespace:
            corr_oracle_templates = set([gt_template.replace(" ", "") for gt_template in corr_oracle_templates])
            identified_template = identified_template.replace(" ", "")
 
        if corr_oracle_templates == {identified_template}:
            if (filter_templates is None) or (identified_template in filter_templates):
                correct_parsing_templates += 1

    if filter_templates is not None:
        PTA = correct_parsing_templates / len(filter_identify_templates)
        RTA = correct_parsing_templates / len(filter_templates)
    else:
        PTA = correct_parsing_templates / len(grouped_df)
        RTA = correct_parsing_templates / len(series_groundtruth_valuecounts)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)
    return FTA, PTA, RTA

def calculate_edit_distance(groundtruth, parsedresult):
    edit_distance_result, normalized_ed_result, cache_dict = [], [] , {}
    iterable = zip(groundtruth['EventTemplate'].values, parsedresult['EventTemplate'].values)
    length_logs = len(groundtruth['EventTemplate'].values)
    for i, j in iterable:
        if i != j:
            if (i, j) in cache_dict:
                ed = cache_dict[(i, j)]
            else:
                if pd.isna(j):
                    ed = len(i)
                    normalized_ed = 1 - ed / len(i)
                else:
                    ed = edit_distance(i, j)
                    normalized_ed = 1 - ed / max(len(i), len(j))
                cache_dict[(i, j)] = ed
            
            edit_distance_result.append(ed)
            normalized_ed_result.append(normalized_ed)

    accuracy_ED = sum(edit_distance_result) / length_logs
    accuracy_NED = (sum(normalized_ed_result) + length_logs - len(normalized_ed_result)) / length_logs
    return accuracy_ED, accuracy_NED

def evaluate_metrics(dataset_name, groundtruth_path, result_path, limit=2000, no_whitespace=False):
    """
    Evaluate various metrics for log parsing results.
    
    Parameters:
    dataset_name (str): Name of the dataset.
    groundtruth_path (str): Path to the ground truth log file. CSV file with columns "EventTemplate" and "Content".
    result_path (str): Path to the parsed result log file. CSV file with columns "EventTemplate" and "Content".
    
    Returns:
    df (pd.DataFrame): DataFrame with the evaluation metrics.
    """
    df = pd.DataFrame(columns=['Dataset', 'PA', 'PTA', 'RTA', "FTA", 'GA', "ED", "NED"])
    df_groundtruth = pd.read_csv(groundtruth_path).iloc[:limit]
    # some log contents in corrected version are not identical to the non corrected contents!!!
    if "_corrected" in groundtruth_path:
        df_groundtruth_noncorrected = pd.read_csv(groundtruth_path.replace("_corrected", "")).iloc[:limit]
        df_groundtruth["Content"] = df_groundtruth_noncorrected["Content"] # exchange with non corrected version
    if not os.path.exists(result_path):
        return None
    df_parsedlog = pd.read_csv(result_path).iloc[:limit]
    if no_whitespace:
        PA = evaluatePA_no_whitespace(df_groundtruth, df_parsedlog)
    else:
        PA = evaluatePA(df_groundtruth, df_parsedlog)
    FTA, PTA, RTA = evaluate_template_level(df_groundtruth, df_parsedlog, no_whitespace=no_whitespace)
    GA = evaluateGA(df_groundtruth, df_parsedlog)
    ED, NED = calculate_edit_distance(df_groundtruth, df_parsedlog)
    if dataset_name not in df['Dataset'].values:
        df.loc[len(df)] = [dataset_name, PA, PTA, RTA, FTA, GA, ED, NED]
    else:
        df.loc[df['Dataset'] == dataset_name, 'PA'] = PA
        df.loc[df['Dataset'] == dataset_name, 'PTA'] = PTA
        df.loc[df['Dataset'] == dataset_name, 'RTA'] = RTA
        df.loc[df['Dataset'] == dataset_name, 'FTA'] = FTA
        df.loc[df['Dataset'] == dataset_name, 'GA'] = GA
        df.loc[df['Dataset'] == dataset_name, 'ED'] = ED
        df.loc[df['Dataset'] == dataset_name, 'NED'] = NED
    df = df.round(3)
    return df
