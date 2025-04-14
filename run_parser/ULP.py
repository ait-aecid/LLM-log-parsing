from utils.parser_utils import *

def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"

    os.chdir(PROJECT_FOLDER + 'parsers/logparser/logparser')
    sys.path.append(PROJECT_FOLDER + 'parsers/logparser/logparser')
    from ULP import LogParser

    setting = settings[dataset]
    log_format = dataset_format
    regex = setting["regex"]
    time_start = time.time()
    parser = LogParser(log_format, indir=os.path.join(in_dir, dataset), outdir=out_dir, rex=regex)
    parser.parse(f"{dataset}_{dataset_type}.log")
    runtime = time.time() - time_start

    if dataset_type == "full":
        return runtime, 0.0, 0.0
    return runtime, 0.0