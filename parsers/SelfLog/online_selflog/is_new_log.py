import csv
import os
import re

def csv_to_dict(file_path):
    result_dict = {}
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            print(f"{file_path} does not exist, creating it.")
    else:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 2:
                    key = row[0]
                    value = row[1]
                    result_dict[key] = value
    return result_dict

def dict_to_csv(data_dict, file_path):
    with open(file_path, mode='w+', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for key, value in data_dict.items():
            csv_writer.writerow([key, value])

def belongs_exsist_template(groups, group_tokens):

    groups_keys = groups.keys()
    exsists = {}
    for groups_key in groups_keys:
        if groups_key in group_tokens.keys():
            exsists[group_tokens[groups_key]] = groups[groups_key]
            groups[groups_key] = []

    return groups, exsists




