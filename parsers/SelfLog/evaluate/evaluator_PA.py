import re

import pandas as pd

# from b  import benchmark_settings
import os
input_dir = os.getcwd() + '/logs/'
output_dir = os.getcwd() + '/evaluate/PA_result/'

log_templateID = [
    "HDFS/HDFS_2k.log_parsed.csv",
    "Hadoop/Hadoop_2k.log_parsed.csv",
    "Spark/Spark_2k.log_parsed.csv",
    "Zookeeper/Zookeeper_2k.log_parsed.csv",
    "BGL/BGL_2k.log_parsed.csv",
    "HPC/HPC_2k.log_parsed.csv",
    "Thunderbird/Thunderbird_2k.log_parsed.csv",
    "Windows/Windows_2k.log_parsed.csv",
    "Linux/Linux_2k.log_parsed.csv",
    "Andriod/Andriod_2k.log_parsed.csv",
    "HealthApp/HealthApp_2k.log_parsed.csv",
    "Apache/Apache_2k.log_parsed.csv",
    "Proxifier/Proxifier_2k.log_parsed.csv",
    "OpenSSH/OpenSSH_2k.log_parsed.csv",
    "OpenStack/OpenStack_2k.log_parsed.csv",
    "Mac/Mac_2k.log_parsed.csv"
]

log_template = [
    "HDFS/HDFS_parsed_templates.csv",
    "Hadoop/Hadoop_parsed_templates.csv",
    "Spark/Spark_parsed_templates.csv",
    "Zookeeper/Zookeeper_parsed_templates.csv",
    "BGL/BGL_parsed_templates.csv",
    "HPC/HPC_parsed_templates.csv",
    "Thunderbird/Thunderbird_parsed_templates.csv",
    "Windows/Windows_parsed_templates.csv",
    "Linux/Linux_parsed_templates.csv",
    "Andriod/Andriod_parsed_templates.csv",
    "HealthApp/HealthApp_parsed_templates.csv",
    "Apache/Apache_parsed_templates.csv",
    "Proxifier/Proxifier_parsed_templates.csv",
    "OpenSSH/OpenSSH_parsed_templates.csv",
    "OpenStack/OpenStack_parsed_templates.csv",
    "Mac/Mac_parsed_templates.csv"
]

groundtruth_path = [
    "HDFS/HDFS_2k.log_structured.csv",
    "Hadoop/Hadoop_2k.log_structured.csv",
    "Spark/Spark_2k.log_structured.csv",
    "Zookeeper/Zookeeper_2k.log_structured.csv",
    "BGL/BGL_2k.log_structured.csv",
    "HPC/HPC_2k.log_structured.csv",
    "Thunderbird/Thunderbird_2k.log_structured.csv",
    "Windows/Windows_2k.log_structured.csv",
    "Linux/Linux_2k.log_structured.csv",
    "Andriod/Andriod_2k.log_structured.csv",
    "HealthApp/HealthApp_2k.log_structured.csv",
    "Apache/Apache_2k.log_structured.csv",
    "Proxifier/Proxifier_2k.log_structured.csv",
    "OpenSSH/OpenSSH_2k.log_structured.csv",
    "OpenStack/OpenStack_2k.log_structured.csv",
    "Mac/Mac_2k.log_structured.csv"
]

def correct_lstm(groudtruth, parsedresult):
    delimiters = [' ']
    pattern = '|'.join(map(re.escape, delimiters))
    try:
        parsedresult = parsedresult.strip('"')
    except:
        parsedresult = str(parsedresult)
    tokens1 = re.split(pattern, groudtruth)
    tokens2 = re.split(pattern, parsedresult)
    tokens1 = ["<*>" if "<*>" in token else token for token in tokens1]
    tokens2 = ["<*>" if "<*>" in token else token for token in tokens2]
    return list(filter(lambda x: x != "", tokens1)) == list(filter(lambda x: x != "", tokens2))

def calculate_parsing_accuracy_lstm(groundtruth_df, parsedresult_df, filter_templates=None):

    error_templateID = []
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]
    groundtruth_templates = list(groundtruth_df['EventTemplate'])
    parsedresult_templates = list(parsedresult_df['template'])
    correct_tempalte = []
    correctly_parsed_messages = 0
    for i in range(len(groundtruth_templates)):
        if correct_lstm(groundtruth_templates[i], parsedresult_templates[i]):
            correctly_parsed_messages += 1
            if groundtruth_templates[i] not in correct_tempalte:
                correct_tempalte.append(groundtruth_templates[i])
        else:
            if groundtruth_df["EventId"][i] not in error_templateID:
                error_templateID.append(groundtruth_df["EventId"][i])
                # print(groundtruth_templates[i])
                # print(parsedresult_templates[i])

    PA = float(correctly_parsed_messages) / len(groundtruth_templates)
    PTA = len(correct_tempalte) / len(parsedresult_df['template'].unique())
    RTA = len(correct_tempalte) / len(groundtruth_df['EventTemplate'].unique())

    print('Parsing_Accuracy (PA): {:.4f} (PTA): {:.4f} (RTA): {:.4f}'.format(PA, PTA, RTA))
    return PA,PTA,RTA

def evaluate_PA():
    for i in range(len(log_templateID)):
        dict_template = {}
        template_file = os.path.join(input_dir, log_template[i])
        df = pd.read_csv(template_file, header=None, skiprows=1)
        dict_template = df.set_index(0).squeeze().to_dict()
        parsed_file = pd.read_csv(os.path.join(input_dir, log_templateID[i])) #, nrows=200
        parsed_file['template'] = parsed_file['EventId'].map(dict_template)
        parsed_file = pd.concat([parsed_file, pd.DataFrame(parsed_file['EventId'].map(dict_template), columns=['template'])], sort=False)
        parsed_file.to_csv(os.path.join(output_dir, os.path.basename(log_templateID[i])), index=False, quoting=0)
    PA_toatal = []
    PTA_toatal = []
    RTA_toatal = []
    for i in range(len(log_templateID)):
        parsed_file = pd.read_csv(os.path.join(output_dir, os.path.basename(log_templateID[i])))
        groundtruth_file = pd.read_csv(os.path.join(input_dir, groundtruth_path[i])) #, nrows=200
        PA,PTA,RTA = calculate_parsing_accuracy_lstm(groundtruth_file, parsed_file, None)
        PA_toatal.append(PA)
        PTA_toatal.append(PTA)
        RTA_toatal.append(RTA)
    PA_average = sum(PA_toatal) / len(PA_toatal)
    PTA_average = sum(PTA_toatal) / len(PTA_toatal)
    RTA_average = sum(RTA_toatal) / len(RTA_toatal)
    for i in range(len(PA_toatal)):
        print(os.path.dirname(log_templateID[i]), PA_toatal[i], PTA_toatal[i], RTA_toatal[i])
    print('Average_Parsing_Accuracy (PA): {:.4f} (PTA): {:.4f} (RTA): {:.4f} '.format(PA_average,PTA_average,RTA_average))
    return PA_toatal, PTA_toatal, RTA_toatal


