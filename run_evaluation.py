from utils.parser_utils import *
# from parser_run import *
from utils.evaluate import evaluate_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

parsers = [
    # non-LLM parsers
    # "Drain",
    # "ULP",
    # "Brain",
    # "Logram",
    # "AEL",
    # unsupervised parsers
    "LLM_TD",
    "LogBatcher",
    "LogPrompt",
    "OpenLogParser",
    "SelfLog",
    # supervised parsers
    "DivLog-2",
    "DivLog-4",
    "LILAC-4",
    "LILAC-2",
]

non_llm_parsers = sorted(["Drain", "Brain", "ULP", "AEL", "SPELL"]) #"Logram", 

datasets = [
    'Android',
    'Apache',
    'BGL',
    'HDFS',
    'HPC',
    'Hadoop',
    'HealthApp',
    'Linux',
    'Mac',
    'OpenSSH',
    'OpenStack',
    'Proxifier',
    'Spark',
    'Thunderbird',
    'Windows',
    'Zookeeper',
    "Audit" # custom
]

total_runs = 3

params = {
    "in_dir": DATA_FOLDER + "2k/",
    "dataset_type": "2k",
    "log_format": True,
}

def evaluate(model, multiple_runs_list, datasets, parsers, corrected_LogHub=True):
    str_add = "_corrected_LogHub" if corrected_LogHub else "_LogHub"
    limit = 2000
    for dataset in datasets:
        input_dir = params["in_dir"]
        for parser in parsers:
            print(f"--- {dataset} - {parser}", flush=True)
            corrected_str = "_corrected" if corrected_LogHub else ""
            groundtruth_path = os.path.join(input_dir, dataset, f"{dataset}_{params["dataset_type"]}.log_structured{corrected_str}.csv")
            results_runs = []
            for run in range(1,total_runs+1):
                run_dir = f"run{run}" if parser in multiple_runs_list else ""
                out_dir = os.path.join(OUTPUT_FOLDER, model, parser, run_dir)
                result_path = os.path.join(out_dir, f"{dataset}_{params["dataset_type"]}.log_structured.csv")
                #result_path = os.path.join(OUTPUT_FOLDER, parser, f"{dataset}_{params["dataset_type"]}.log_structured.csv")
                if not os.path.exists(result_path):
                    print("Path doesn't exist:", result_path)
                    continue
                df_result = evaluate_metrics(dataset, groundtruth_path, result_path, limit=limit)
                # display(df_result)
                if df_result is None:
                    print("Skipping")
                    continue
                summary_path = os.path.join(out_dir, f"_summary{str_add}.csv")
                if not os.path.exists(summary_path):
                    df_result.to_csv(summary_path, index=False)
                else:
                    df_result.to_csv(summary_path, index=False, mode="a", header=False)
                
                if parser not in multiple_runs_list:
                    break
                results_runs.append(df_result.set_index("Dataset"))

            if parser not in multiple_runs_list:
                continue
            df_results_runs = reduce(lambda x, y: x.add(y, fill_value=0), results_runs)/total_runs
            summary_path = os.path.join(OUTPUT_FOLDER, model, parser, f"_summary{str_add}.csv")
            if not os.path.exists(summary_path):
                df_results_runs.to_csv(summary_path, index=True)
            else:
                df_results_runs.to_csv(summary_path, index=True, mode="a", header=False)

def get_results(model, parsers, corrected_LogHub=True, exclude=[]):
    parsers_list = [p for p in parsers if p not in exclude]
    dfs = []
    str_add = "_corrected_LogHub" if corrected_LogHub else "_LogHub"
    for i, parser in enumerate(parsers_list):
        path = os.path.join(OUTPUT_FOLDER, model, parser, f"_summary{str_add}.csv")
        try:
            df_res = pd.read_csv(path).drop_duplicates(subset=["Dataset"], keep="last").set_index("Dataset")
            dfs.append(df_res[["GA","FTA","PA","NED"]])
        except Exception as e:
            print(f"Error: {e}")
            continue
    df_full = pd.concat(dfs, axis=1, keys=parsers_list)
    #pd.set_option("display.max_columns", None)
    #display(df_full.round(3))
    return df_full

def plot(df, parsers, exclude=[], save_path=None, ylim=(-0.02,1.02)):
    #parsers_list = [p for p in parsers if p not in exclude]
    dataframes = [df[p] if p not in exclude else df[parsers[0]].map(lambda x: -1) for p in parsers]

    plt.rcParams.update({'font.size': 8.5})
    fig, axes = plt.subplots(nrows=1, ncols=len(dataframes), figsize=(11/9*(len(dataframes)), 2.5), sharey=True)
    for ax, df_, title in zip(axes, dataframes, parsers):
        sns.boxplot(data=df_, ax=ax)
        ax.set_title(title)
    
    plt.ylim(ylim)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    

def plot_bar(df, parsers, exclude=[], width=11, height=3, save_path=None):
    parsers_list = [p for p in parsers if p not in exclude]
    dataframes = [df[p] for p in parsers_list]
    plt.rcParams.update({'font.size': 8.5})
    fig, axes = plt.subplots(nrows=1, ncols=len(dataframes), figsize=(width/9*len(parsers_list), height), sharey=True)
    for ax, df_, title in zip(axes, dataframes, parsers_list):
        sns.barplot(data=df_, ax=ax)
        ax.set_title(title)
    #plt.ylim((-0.02,1.02))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    

def plot_bar_stacked(df, parsers, exclude=[], width=11, height=3, save_path=None):
    parsers_list = [p for p in parsers if p not in exclude]
    dataframes = [df[p] for p in parsers_list]
    plt.rcParams.update({'font.size': 8.5})
    n_cols = len(dataframes) // 2 + len(dataframes) % 2  # Calculate number of columns for two rows
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(width/9*n_cols, height * 2), sharey=True)
    axes = axes.flatten()  # Flatten axes for easier iteration
    for ax, df_, title in zip(axes, dataframes, parsers_list):
        sns.barplot(data=df_, ax=ax)
        ax.set_title(title)
    # Hide any unused subplots
    for ax in axes[len(dataframes):]:
        ax.axis('off')
    plt.ylim((-0.02, 1.02))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    

def get_time_eval(df, parsers, sort_by="Computation Time"):
    if sum(df["run"] != 2) != 0:
        print("Number of runs not equal to 3!")  # safety check
    total_mean = [df.loc[p]["total_runtime"].mean() for p in parsers]
    total_std = [df.loc[p]["total_runtime"].std() for p in parsers]
    invoc_mean = [df.loc[p]["invocation_time"].mean() for p in parsers]
    invoc_std = [df.loc[p]["invocation_time"].std() for p in parsers]
    comp_mean = [(df.loc[p]["total_runtime"] - df.loc[p]["invocation_time"]).mean() for p in parsers]
    comp_std = [(df.loc[p]["total_runtime"] - df.loc[p]["invocation_time"]).std() for p in parsers]
    columns = pd.MultiIndex.from_tuples([
        ('Computation Time', r'$\mu$'),
        ('Computation Time', r'$\sigma$'),
        ('Invocation Time', r'$\mu$'),
        ('Invocation Time', r'$\sigma$'),
        ('Total Runtime', r'$\mu$'),
        ('Total Runtime', r'$\sigma$'),
    ])
    data = [[comp_mean[i], comp_std[i], invoc_mean[i], invoc_std[i], total_mean[i], total_std[i]] for i in range(len(total_mean))]
    time_df = pd.DataFrame(data, columns=columns, index=parsers)
    return time_df.sort_values(by=(sort_by, r'$\mu$'), ascending=False).round(2)

def get_time_table(model, parsers, datasets, folder=None, ):
    if folder:
        df_time_raw = pd.read_csv(os.path.join(folder, model, "times.csv"))
    else:
        df_time_raw = pd.read_csv(os.path.join(OUTPUT_FOLDER, model, "times.csv"))
    df_time_raw = df_time_raw.sort_values(by=["parser","dataset","run"])
    df_time = df_time_raw[df_time_raw['dataset'] != 'Audit-light']#.groupby(['parser', "dataset"]).tail(3)
    tmp_list = []
    # use for loop because groupby does weird things ...
    for parser in parsers:
        for dataset in datasets:
            for run in range(1, total_runs + 1):
                tmp = df_time[df_time["parser"] == parser]
                tmp = tmp[df_time["dataset"] == dataset]
                tmp = tmp[df_time["run"] == run].tail(1)
                tmp_list.append(tmp)
    df_time_avg = pd.concat(tmp_list)
    df_time_avg = df_time_avg.groupby(['parser', "dataset"])[["run","total_runtime","invocation_time","n_queries"]].sum().apply(lambda x: x/total_runs)
    return df_time_avg

def check_decimal(x):
    if len(x) != 4:
        return x + "0"
    else:
        return x
    
# plot results per LLM
def extract(df):
    parsers_list = [p for p in parsers if p not in ["LogPrompt"]] # exclude LogPrompt
    return sum([df[p] for p in parsers_list])/len(parsers_list)

evaluate("no-LLM", non_llm_parsers, datasets, non_llm_parsers, corrected_LogHub=True)
df_no_llm = get_results("no-LLM", non_llm_parsers, corrected_LogHub=True)
plot(df_no_llm, non_llm_parsers, save_path="plots/no_llm.pdf")

evaluate("codellama:7b-instruct", parsers, datasets, parsers, corrected_LogHub=True)
df_codellama = get_results("codellama:7b-instruct", parsers, corrected_LogHub=True)
evaluate("gpt-3.5-turbo", parsers, datasets, parsers, corrected_LogHub=True)
df_gpt = get_results("gpt-3.5-turbo", parsers, corrected_LogHub=True)
evaluate("gpt-3.5-turbo_LogHub", parsers, datasets, parsers, corrected_LogHub=False)
df_gpt_uncorrected = get_results("gpt-3.5-turbo_LogHub", parsers, corrected_LogHub=False)
p_list = parsers.copy()
p_list.remove("LogPrompt")
evaluate("deepseek-ai/DeepSeek-R1", p_list, datasets, p_list, corrected_LogHub=True)
df_deepseek = get_results("deepseek-ai/DeepSeek-R1", parsers, exclude=["LogPrompt"], corrected_LogHub=True)

plot(df_codellama, parsers, save_path="plots/codellama.pdf")
plot(df_gpt, parsers, save_path="plots/gpt.pdf")
# plot(df_gpt_uncorrected, parsers, save_path="plots/gpt_uncorrected.pdf")
plot(df_deepseek, parsers, exclude=["LogPrompt"], save_path="plots/deepseek.pdf")

df_diff = (df_gpt.drop(index="Audit")-df_gpt_uncorrected.drop(index="Audit"))
plot(df_diff, parsers, save_path="plots/gpt_minus_gpt_uncorrected.pdf", ylim=(-1.1,1.1))

df_gpt_incl_non_llm_parser = pd.concat([df_no_llm, df_gpt], axis=1)
sorted_columns = df_gpt_incl_non_llm_parser.loc["Audit"].loc[:,"GA"].sort_values(ascending=False).index
df_sorted = df_gpt_incl_non_llm_parser[sorted_columns]
plot_bar_stacked(pd.DataFrame(df_sorted.loc["Audit"]).T, sorted_columns, width=10, height=2, save_path="plots/gpt_incl_no_llm_audit.pdf")

# # print table with best performances in bold
# tmp = df_gpt_incl_non_llm_parser.T.copy()
# tmp["Average"] = tmp.mean(axis=1).round(2)
# table = tmp.T.round(2).copy()

# for d in datasets + ["Average"]:
#     for m in ["GA", "FTA", "PA", "NED"]:
#         values = {p: table.loc[d, (p, m)] for p in non_llm_parsers + parsers}
#         max_value = max(values.values())
#         best_parsers = [p for p, v in values.items() if v == max_value]

#         for best_parser in best_parsers:
#             table.loc[d, (best_parser, m)] = r"\textbf{" + check_decimal(str(table.round(2).loc[d, (best_parser, m)])) + r"}"

# # print to latex
# tab = table.T
# tab = tab.rename(index={c: r"\textbf{" + str(c) + r"}" for c in tab.index})
# tab = tab.rename(columns={c: r"\begin{turn}{90}{" + str(c) + r"}\end{turn}" for c in tab.columns})
# print(tab.to_latex(float_format="%.2f"))

dfs = {"CodeLlama": extract(df_codellama), "GPT-3.5": extract(df_gpt), "DeepSeek R1": extract(df_deepseek)}
plot(dfs, dfs.keys(), save_path="plots/all_models_aggregated.pdf")

# prepare plots for efficiency evaluation
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index")

runtime_parsers = [
    "Drain",
    "ULP",
    "Brain",
    "SPELL",
    "AEL",
    "OpenLogParser",
    # "LogPrompt",
    "LLM_TD",
    "LogBatcher",
    # "SelfLog", # SelfLog got killed
    "LILAC-2",
    "LILAC-4",
]

df_time_bgl = get_time_table("gpt-3.5-turbo", runtime_parsers, datasets=["BGL"], folder="output-full/")
time_table_bgl = get_time_eval(df_time_bgl, parsers=runtime_parsers, sort_by="Computation Time")
df_time_hdfs = get_time_table("gpt-3.5-turbo", runtime_parsers, datasets=["HDFS"], folder="output-full/")
time_table_hdfs = get_time_eval(df_time_hdfs, parsers=runtime_parsers, sort_by="Computation Time")

print("Plots procuded!")