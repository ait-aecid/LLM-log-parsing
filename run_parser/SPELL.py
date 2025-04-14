from utils.parser_utils import *
from utils.settings import SPELL_settings

def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"

    os.chdir(PROJECT_FOLDER + 'parsers/logparser/logparser')
    sys.path.append(PROJECT_FOLDER + 'parsers/logparser/logparser')
    from Spell import LogParser

    setting = settings[dataset]
    log_format = dataset_format
    tau = SPELL_settings[dataset]["tau"]
    regex = setting["regex"]
    time_start = time.time()
    parser = LogParser(indir=os.path.join(in_dir, dataset), outdir=out_dir, log_format=log_format, tau=tau, rex=regex)
    parser.parse(f"{dataset}_{dataset_type}.log")
    runtime = time.time() - time_start

    if dataset_type == "full":
        return runtime, 0.0, 0.0
    return runtime, 0.0