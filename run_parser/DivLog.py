# partially from DivLog/demo.py

from utils.parser_utils import *

def DivLog_setup(input_dir, dataset):
    import os
    #os.system("pip install openai==0.28.1")
    sys.path.append(PROJECT_FOLDER + 'parsers/logparser/logparser/DivLog')
    os.chdir(PROJECT_FOLDER + 'parsers/logparser/logparser/DivLog')
    import openai
    from DivLog import ModelParser
    import json
    import os
    import pandas as pd
    from tqdm import tqdm
    from openai import OpenAI
    output_dir = "embeddings/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(output_dir + dataset + ".json") == True:
       return ModelParser
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    client = OpenAI()
    embedding = dict()
    print("Embedding " + dataset + "...")
    i = pd.read_csv(input_dir + '/' + dataset + '/' + dataset + "_2k.log_structured.csv")
    contents = i['Content']
    for log in tqdm(contents):
        response = client.embeddings.create(input=log, model='text-embedding-ada-002').data[0].embedding
        embedding[log] = response
    o = json.dumps(embedding, separators=(',',':'))
    with open(output_dir + dataset + ".json","w") as f:
        f.write(o)
        f.close()
    return ModelParser


def parse(in_dir, out_dir, settings, dataset="Audit", dataset_type="2k", model="gpt-3.5-turbo", log_format=True, n_candidates=4, run=None, corrected_LH=None,
          
    # other params
    map_path='maps',
    emb_path='embeddings',
    split_method='custom',
    order_method='KNN',
    permutation='ascend',
    warmup=False,
    model_name='gptC',
    limit=2000,
    N=5,
    subname='',
    evaluate=False
):
    # if log_format:
    #     dataset_format = settings[dataset]["log_format"]
    # else:
    #     dataset_format = "<Content>"
    
    ModelParser = DivLog_setup(input_dir=in_dir, dataset=dataset)
    print("Parsing " + dataset + " ...")
    if not os.path.exists(map_path):
        os.mkdir(map_path)
    if not os.path.exists(emb_path):
        print("Embedding path does not exist. Please check the path.")
        exit()

    #cand_ratio = n_candidates / limit # 2000 for loghub-2k

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    parser = ModelParser(
        log_path= in_dir,        # .log_structured_corrected.csv !!! corrected version !!!
        result_path = out_dir,
        map_path = map_path,          # .map_json
        dataset = dataset,             # 16 datasets
        emb_path = emb_path,           # embedding
        cand_ratio = n_candidates,     # number of candidates ### CUSTOM
        split_method = split_method,   # random or DPP
        order_method = order_method,   # random or KNN
        permutation = permutation,     # permutation
        warmup = warmup,               # warmup or not
        subname = subname,             # subname of the files
        evaluate = evaluate,           # evaluate or not
        corrected_LH = corrected_LH ### CUSTOM
    )
    time_start = time.time()
    parser.BatchParse(model = model, 
        model_name = model_name, 
        limit = limit,         # number of logs for testing
        N = N,                  # number of examples in the prompt
    )
    runtime = time.time() - time_start
    return runtime, parser.invoc_time
    # # create template file
    # out_file_structured = os.path.join(out_dir, "DivLog", f"{dataset}_{dataset_type}." + 'log_structured.csv')
    # df = pd.read_csv(out_file_structured)[["EventId", "EventTemplate"]].drop_duplicates()
    # df.to_csv(os.path.join(out_dir, "DivLog", f"{dataset}_{dataset_type}.log_templates.csv"))
