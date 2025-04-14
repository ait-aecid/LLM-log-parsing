from utils.parser_utils import *

# load api key, dataset format and parser
def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None):
    if log_format:
        dataset_format = settings[dataset]["log_format"]
    else:
        dataset_format = "<Content>"

    os.chdir(PROJECT_FOLDER + 'parsers/LogBatcher') # run in LogBatcher directory
    sys.path.append(PROJECT_FOLDER + 'parsers/LogBatcher')
    # partially from LogBatcher/logbatcher/demo.py:
    from logbatcher.parsing_base import single_dataset_paring
    from logbatcher.parser import Parser
    from logbatcher.util import data_loader
    
    settings.update({"api_key_from_openai": OPENAI_KEY})
    settings.update({"api_key_from_together": TOGETHER_API_KEY}) #, "api_key_from_together": ""
    parser = Parser(model, out_dir, settings)
    # load contents from raw log file, structured log file or content list
    contents = data_loader(
        file_name=f"{in_dir}/{dataset}/{dataset}_{dataset_type}.log",
        dataset_format= dataset_format,
        file_format ='raw'
    )
    time_start = time.time()
    # parse logs
    single_dataset_paring(
        dataset=dataset,
        contents=contents,
        output_dir= out_dir + "/",
        parser=parser,
        debug=False,
        batch_size=10,
        data_type=dataset_type
    )
    time_end = time.time()
    runtime = time_end - time_start
    if dataset_type == "full":
        return runtime, parser.time_consumption_llm, parser.n_queries
    return runtime, parser.time_consumption_llm