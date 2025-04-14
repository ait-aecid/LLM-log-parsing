from utils.parser_utils import *
from utils.settings import brain_settings

def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"

    os.chdir(PROJECT_FOLDER + 'parsers/logparser/logparser')
    sys.path.append(PROJECT_FOLDER + 'parsers/logparser/logparser')
    from Brain import LogParser

    setting = settings[dataset]
    log_format = dataset_format
    delimiter = brain_settings[dataset]["delimiter"]
    threshold = brain_settings[dataset]["theshold"]
    regex = setting["regex"]
    time_start = time.time()
    parser = LogParser(logname=dataset, log_format=log_format, indir=os.path.join(in_dir, dataset), outdir=out_dir, threshold=threshold, delimeter=delimiter, rex=regex)
    parser.parse(f"{dataset}_{dataset_type}.log")
    runtime = time.time() - time_start

    if dataset_type == "full":
        return runtime, 0.0, 0.0
    return runtime, 0.0