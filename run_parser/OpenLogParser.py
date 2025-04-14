from utils.parser_utils import *

def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"

    import os
    import sys
    os.chdir(PROJECT_FOLDER + 'parsers/LibreLog/parser') # run in LogBatcher directory
    sys.path.append(PROJECT_FOLDER + 'parsers/LibreLog/parser')

    import grouping
    import csv
    import llama_parser
    import regex as re
    import pandas as pd
    import regex_manager
    from tqdm import tqdm
    from pathlib import Path

    def read_column_from_csv(file_path, column_name="Content"):
        column_data = []
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    column_data.append(row[column_name])
                else:
                    raise ValueError(
                        f"The column '{column_name}' does not exist in the CSV file."
                    )
        return column_data


    def group_logs_using_parser(grouped_logs):
        df = pd.DataFrame(grouped_logs, columns=["Content", "EventId", "EventTemplate"])
        df = df[["Content", "EventId", "EventTemplate"]]
        grouped = df.groupby("EventId")
        groups_dict = {}
        for name, group in grouped:
            groups_dict[name] = group.to_dict("records")
        return groups_dict


    def get_logs_from_group(group_list):
        logs_from_group = []
        for ele in group_list:
            logs_from_group.append(ele["Content"])
        return logs_from_group


    def sort_dict_by_content_length(input_dict):
        def count_words_in_content(entry):
            return len(entry["Content"].split())

        sorted_items = sorted(
            input_dict.items(), key=lambda item: count_words_in_content(item[1][0])
        )

        sorted_dict = {key: value for key, value in sorted_items}
        return sorted_dict


    def append_unique_to_csv(data_list, file_path):
        new_data = pd.DataFrame(data_list)
        file = Path(file_path)

        if "Count" in new_data.columns:
            new_data = new_data.drop(columns="Count")
        new_data = new_data.groupby(new_data.columns.tolist(), as_index=False).size()
        new_data = new_data.rename(columns={"size": "Count"})

        if file.is_file():
            existing_data = pd.read_csv(file_path, dtype={1: str})
        else:
            existing_data = pd.DataFrame(columns=new_data.columns)

        combined_data = pd.concat([existing_data, new_data], ignore_index=True)

        combined_data.to_csv(file_path, index=False, header=True)
        return file_path

    def regex_to_template(regex_pattern):
        # Replace non-greedy capture groups (.*?) with <*>
        template = re.sub(r'\(\.\*\?\)\$', '<*>', regex_pattern)
        template = re.sub(r'\(\.\*\?\)', '<*>', template)
        template = re.sub(r'\\(.)', r'\1', template)
        return template

    print(f"Start Parsing {dataset}", flush=True)
    log_file = f"{in_dir}/{dataset}/{dataset}_{dataset_type}.log_structured.csv"
    path_prefix = "../results_offline_similar/"
    out_path = f"{path_prefix}{dataset}/"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    setting = settings[dataset]

    t1 = time.time()

    Drain_parser1 = grouping.LogParser(
        rex=setting["regex"], depth=setting["depth"], st=setting["st"]
    )
    logs = read_column_from_csv(log_file)
    grouped_logs = Drain_parser1.parse(logs)
    groups_dict = group_logs_using_parser(grouped_logs)
    groups_dict = sort_dict_by_content_length(groups_dict)

    regex_manager1 = regex_manager.RegexTemplateManager()
    llama_parser1 = llama_parser.LogParser(
        pipeline=get_client(model), 
        model=model,
        regex_manager1=regex_manager1,
        regex_sample=5,
    )
    results = []
    for eventid in tqdm(groups_dict.keys(), desc=f"Processing events {dataset}"):
        append_unique_to_csv(groups_dict[eventid], out_path + "group.csv")
        res_list = []
        logs_from_group = get_logs_from_group(groups_dict[eventid])
        res_list = llama_parser1.parse(groups_dict[eventid], logs_from_group)
        results = results + res_list
    
    t2 = time.time()

    df = pd.DataFrame(results, columns=["Content", "EventId", "EventTemplate"])
    df["EventTemplate"] = df["EventTemplate"].apply(lambda x: regex_to_template(x))
    out_structured = os.path.join(out_dir, f"{dataset}_{dataset_type}." + 'log_structured.csv')
    df.to_csv(out_structured)
    out_templates = os.path.join(out_dir, f"{dataset}_{dataset_type}." + 'log_templates.csv')
    df.drop_duplicates(subset=["EventTemplate"])[["EventId", "EventTemplate"]].to_csv(out_templates)

    if dataset_type == "full":
        return t2-t1, llama_parser1.invoc_time, llama_parser1.n_queries

    return t2-t1, llama_parser1.invoc_time