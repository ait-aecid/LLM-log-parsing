# partially from 

from utils.parser_utils import *

def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"
    # sample data for LILAC before each run
    if corrected_LH:
        groundtruth_paths = get_data_paths(in_dir, identifier="structured_corrected.csv")
    else:
        groundtruth_paths = get_data_paths(in_dir, identifier="log_structured.csv")
    df_sample = sample_data([p for p in groundtruth_paths if dataset in p][0], dataset, n_candidates, corrected_LH=corrected_LH)
    samples_lilac = [{"query": df_sample.iloc[i]["Content"], "answer": df_sample.iloc[i]["EventTemplate"].replace("<*>", r"{variables}")} for i in range(len(df_sample))]
    lilac_dir= f"{PROJECT_FOLDER}/parsers/LILAC/full_dataset/sampled_examples/{dataset}"
    if not os.path.exists(lilac_dir):
        os.makedirs(lilac_dir)
    with open(f"{lilac_dir}/{n_candidates}shot.json", "w") as f:
        for s in samples_lilac:
            f.write(json.dumps(s) + "\n")

    os.chdir(PROJECT_FOLDER + 'parsers/LILAC/benchmark/evaluation') # run in LILAC directory
    sys.path.append(PROJECT_FOLDER + 'parsers/LILAC/benchmark/')
    from logparser.LILAC.LILAC import LogParser

    log_file = f"{dataset}/{dataset}_{dataset_type}.log"
    log_file_basename = os.path.basename(log_file)
    # clear temp folder
    os.system(f"rm -r {PROJECT_FOLDER}/parsers/LILAC/temp/*")
    # create output folder
    output_path = out_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    df_sample.to_csv(os.path.join(output_path, f"samples_{dataset}.csv")) # save for reference
    parser = LogParser(
        log_format=dataset_format,
        indir=os.path.join(in_dir, os.path.dirname(log_file)),
        outdir=output_path,
        shot=n_candidates,
        example_size=3,
        model=model,
        data_type=dataset_type,
    )
    time_start = time.time()
    parser.parse(log_file_basename)
    time_end = time.time()
    runtime = time_end - time_start
    if dataset_type == "full":
        return runtime, parser.invoc_time, parser.n_queries
    return runtime, parser.invoc_time