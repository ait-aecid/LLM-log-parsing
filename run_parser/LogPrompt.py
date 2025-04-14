from utils.parser_utils import *

def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    os.chdir(PROJECT_FOLDER + 'parsers/LogPrompt')
    #sys.path.append(PROJECT_FOLDER + 'parsers/LogPrompt')
    if not log_format:
        data_path = os.path.join(in_dir, dataset, dataset + "_" + dataset_type + ".log")
    else:
        data_path = os.path.join(in_dir, dataset, dataset + "_" + dataset_type + ".log_structured.csv")
    # prompt_candidates.txt was altered to only include prompt 2
    command = f"python3 LogPrompt_code.py --API_KEY {get_api_key(model)} --dataset {data_path} --strategy Self --model {model}"
    time_start = time.time()
    os.system(command)
    time_end = time.time()
    runtime = time_end - time_start

    with open("logprompt_time.txt", "r") as f:
        invoc_time = float(f.read().strip())
        f.close()

    df = pd.read_csv("result.csv")
    templates = list(set(df["EventTemplate"]))
    templates_dict = {templates[i]: f"E{i+1}" for i in range(len(templates))}
    df["EventID"] = df["EventTemplate"].apply(lambda x: templates_dict[x])
    df.to_csv(os.path.join(out_dir, dataset + "_" + dataset_type + ".log_structured.csv"), index=False)
    pd.DataFrame({"EventID": list(templates_dict.values()), "EventTemplate": templates}).to_csv(os.path.join(out_dir, dataset + "_" + dataset_type + ".log_templates.csv"), index=False)
    os.system("rm result.csv")

    return runtime, invoc_time
