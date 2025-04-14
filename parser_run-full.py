from utils.parser_utils import *

from run_parser import LILAC, LogBatcher, DivLog, LogPrompt, SelfLog, OpenLogParser, LLM_TD
from run_parser import Drain, ULP, Brain, SPELL, AEL

parsers = {
    # baseline
    "Drain": Drain,
    "ULP": ULP,
    "Brain": Brain,
    "SPELL": SPELL,
    "AEL": AEL,
    # unsupervised parsers
    "OpenLogParser": OpenLogParser,
    # "LogPrompt": LogPrompt,
    "LLM_TD": LLM_TD,
    "LogBatcher": LogBatcher,
    # supervised parsers
    # "SelfLog": SelfLog, # SelfLog gets killed
    "LILAC-2": LILAC,
    "LILAC-4": LILAC,
    # "DivLog-2": DivLog,
    # "DivLog-4": DivLog,
}

multiple_runs_list = list(parsers.keys())

datasets = [
    # 'Android',
    # 'Apache',
    'BGL',
    'HDFS',
    # 'HPC',
    # 'Hadoop',
    # 'HealthApp',
    # 'Linux',
    # 'Mac',
    # 'OpenSSH',
    # 'OpenStack',
    # 'Proxifier',
    # 'Spark',
    # 'Thunderbird',
    # 'Windows',
    # 'Zookeeper',
    # "Audit" # custom
]

model = "gpt-3.5-turbo" # openai api
# model="deepseek-ai/DeepSeek-R1" # togetherai api
# model = "deepseek-reasoner" # deepseek api
# model = "codellama:7b-instruct" # ollama local api

dataset_type = "full" #"2k"

total_runs = 3

params = {
    #"in_dir": DATA_FOLDER + "2k/",
    "in_dir": DATA_FOLDER + f"{dataset_type}/",
    "settings": settings,
    "dataset_type": dataset_type,
    "model": model,
    "log_format": True,
    "corrected_LH": False ### ATTENTION !!!!!! ###
}

if __name__ == "__main__":
    output_folder = OUTPUT_FOLDER[:-1] + "-full" if dataset_type == "full" else OUTPUT_FOLDER
    times_path = os.path.join(output_folder, model, "times.csv")
    error_log = {}
    for dataset in datasets: # per dataset
        params["dataset"] = dataset
        for parser_name, parser in parsers.items(): # per parser
            runs = 1
            if parser_name in multiple_runs_list:
                runs = total_runs # run supervised parsers multiple times
                try:
                    params["n_candidates"] = int(parser_name[-1])
                except:
                    pass
            for i in range(1, runs+1): # per sampling if supervised
                run_dir = ""
                if parser_name in multiple_runs_list:
                    params["run"] = i
                    run_dir = f"run{i}"
                print(f"Running {parser_name} on {dataset}")
                out_dir = os.path.join(output_folder, model, parser_name, run_dir)
                params["out_dir"] = out_dir
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                try:
                    # skip if already exists
                    # if os.path.exists(os.path.join(out_dir, dataset + "_" + params["dataset_type"] + ".log_structured.csv")):
                    #     print("Already exists. Skipping.", dataset)
                    #     continue

                    if dataset_type == "full":
                        runtime, invoc_time, n_queries = parser.parse(**params)
                    else:
                        runtime, invoc_time = parser.parse(**params)

                    dict_time = {"parser": parser_name, "dataset": dataset, "run": i, "total_runtime":runtime, "invocation_time": invoc_time, "n_queries": n_queries}
                    df_time = pd.DataFrame(dict_time, index=[0])
                    if not os.path.exists(times_path):
                        df_time.to_csv(times_path, index=False)
                    else:
                        df_time.to_csv(times_path, mode="a", header=False, index=False)
                        
                except Exception as e:
                    error_log = f"{parser_name} - {dataset} - run{i}: {e}"
                    with open(os.path.join(PROJECT_FOLDER, "error_log.json"), "a") as f:
                        f.write(str(error_log))
                    print(f"Error in {parser}: {e}")
