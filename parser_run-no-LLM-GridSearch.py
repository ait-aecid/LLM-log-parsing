from utils.parser_utils import *

from run_parser import LILAC, LogBatcher, DivLog, LogPrompt, SelfLog, OpenLogParser, LLM_TD
from run_parser import Drain, ULP, Brain, SPELL, AEL

from utils.evaluate import evaluate_metrics
from itertools import product

def evaluate(dataset, parser, out_dir, corrected_LogHub=True,):
    limit = 2000
    input_dir = params["in_dir"]
    print(f"--- {dataset} - {parser}", flush=True)
    corrected_str = "_corrected" if corrected_LogHub else ""
    groundtruth_path = os.path.join(input_dir, dataset, f"{dataset}_{params["dataset_type"]}.log_structured{corrected_str}.csv")
    result_path = os.path.join(out_dir, f"{dataset}_{params["dataset_type"]}.log_structured.csv")
    #result_path = os.path.join(OUTPUT_FOLDER, parser, f"{dataset}_{params["dataset_type"]}.log_structured.csv")
    if not os.path.exists(result_path):
        print("Path doesn't exist:", result_path)
        raise FileNotFoundError
    df_result = evaluate_metrics(dataset, groundtruth_path, result_path, limit=limit)
    return df_result

parsers = {
    # baseline
    "Drain": Drain,
    # "ULP": ULP,
    "Brain": Brain,
    "SPELL": SPELL,
    "AEL": AEL,
    # unsupervised parsers
    # "OpenLogParser": OpenLogParser,
    # # "LogPrompt": LogPrompt,
    # "LLM_TD": LLM_TD,
    # "LogBatcher": LogBatcher,
    # # supervised parsers
    # "SelfLog": SelfLog,
    # "LILAC-2": LILAC,
    # "LILAC-4": LILAC,
    # "DivLog-2": DivLog,
    # "DivLog-4": DivLog,
}

multiple_runs_list = list(parsers.keys())

datasets = [
    # 'Android',
    # 'Apache',
    # 'BGL',
    # 'HDFS',
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
    "Audit" # custom
]

model = "no-LLM"
# model = "gpt-3.5-turbo" # openai api
# model="deepseek-ai/DeepSeek-R1" # togetherai api
# model = "deepseek-reasoner" # deepseek api
# model = "codellama:7b-instruct" # ollama local api

dataset_type = "2k"

total_runs = 1

gs_params = {
    "Drain": {
        "depth": [4,5,6,7,8,9,10,11,12,13,14],
        "st": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    },
    "Brain": {
        "threshold": [2,3,4,5,6,7,8,9,10],
    },
    "AEL": {
        "minEventCount": [1,2,3,4,5,6,7,8,9,10,11,12],
        "merge_percent": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "SPELL": {
        "tau": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    },
}

params = {
    #"in_dir": DATA_FOLDER + "2k/",
    "in_dir": DATA_FOLDER + f"{dataset_type}/",
    "settings": settings,
    "dataset_type": dataset_type,
    "model": model,
    "log_format": True,
    "corrected_LH": True ### ATTENTION !!!!!! ###
}

if __name__ == "__main__":
    output_folder = OUTPUT_FOLDER[:-1] + "-GridSearch/"
    for dataset in datasets: # per dataset
        params["dataset"] = dataset
        for parser_name, parser in parsers.items(): # per parser
            # Generate grid search combinations for the current parser
            param_grid = gs_params[parser_name]
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())

            gs_results = []

            for i, param_values in enumerate(param_combinations):  # per parameter combination
                param_dict = dict(zip(param_names, param_values))
                if parser_name in ["Drain"]:
                    params["settings"]["Audit"].update(param_dict)
                else: 
                    params["settings"][f"{parser_name}_settings"]["Audit"].update(param_dict)

                run_dir = f"run{"_".join([f"{k}_{v}" for k, v in param_dict.items()])}"
                print(f"Running {parser_name} on {dataset}")
                out_dir = os.path.join(output_folder, model, parser_name, run_dir)
                params["out_dir"] = out_dir
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                runtime, invoc_time = parser.parse(**params)

                df_result = evaluate(dataset, parser_name, out_dir, corrected_LogHub=params["corrected_LH"])
                df_result["params"] = [param_dict]

                print(df_result)

                gs_results.append(df_result)

            gs_results_all = pd.concat(gs_results, ignore_index=True)
            # gs_results_all = gs_results_all.sort_values(by=["GA"], ascending=False)
            gs_results_all.to_csv(os.path.join(output_folder, parser_name + "_gs_results.csv"), index=False)
            print("Grid search results saved to:", os.path.join(output_folder, parser_name + "_gs_results.csv"))

