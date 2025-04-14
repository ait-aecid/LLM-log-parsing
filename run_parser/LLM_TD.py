#cat mylog | sed 's/^/AppName /' >test.log
from utils.parser_utils import *

# load api key, dataset format and parser
def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"

    os.chdir(PROJECT_FOLDER + 'parsers/llm-td')
    file_path = f"{in_dir}/{dataset}/{dataset}_{dataset_type}.log"
    df_structured = logfile_to_dataframe(file_path, dataset_format)
    tmp_input_file = "tmp_input.log"
    tmp_log_content = "tmp_log_content.log"
    if log_format: # if format is given then save and use the log content as input
        with open(tmp_log_content, "w") as f:
            for line in df_structured["Content"]:
                f.write(f"{line}\n")
        # add log identifier in front of each log
        os.system(f"cat {tmp_log_content} | sed 's/^/LOG /' >{tmp_input_file}")
    else:
        os.system(f"cat {file_path} | sed 's/^/LOG /' >{tmp_input_file}")

    with open("invoc_time.txt", "w") as f:
        pass

    # run parser
    run_parser = f"./llm-td.pl --model={model} --logfile={tmp_input_file} --script=./llm-query.sh --regexp='(?<line>(?<program>LOG).+)'"
    time_start = time.time()
    os.system(run_parser)
    time_end = time.time()
    runtime = time_end - time_start

    # extract results
    df = pd.read_csv(f"{PROJECT_FOLDER}parsers/llm-td/log_templates.csv", delimiter=",,,")
    df = df.rename(columns={"Pattern": "EventTemplate"})[["EventTemplate"]]
    df["EventTemplate"] = df["EventTemplate"].apply(lambda x: x[4:]) # remove "LOG "
    df["EventId"] = list(range(len(df["EventTemplate"])))
    out_file = os.path.join(out_dir, f"{dataset}_{dataset_type}.log_templates.csv")
    df.to_csv(out_file)
    
    df_structured["EventTemplate"] = df_structured["Content"].apply(lambda x: match_log_to_template(x, df["EventTemplate"].to_list()))
    out_file_structured = os.path.join(out_dir, f"{dataset}_{dataset_type}." + 'log_structured.csv')
    df_structured.to_csv(out_file_structured)

    with open("invoc_time.txt", "r") as f:
        invoc_times = [float(line.strip()) for line in f if line.strip()]
    invoc_time = sum(invoc_times)
    if dataset_type == "full":
        return runtime, invoc_time, len(invoc_times)
    return runtime, invoc_time