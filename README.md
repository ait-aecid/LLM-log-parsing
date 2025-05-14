# LLMs_for_log_parsing
This is the replication repository for **https://arxiv.org/abs/2504.04877** (arXiv). 29 papers concerning LLM-based log parsing were reviewed, seven of them were used for the benchmark. The systematic overview can be found in the excel sheet [categories_clean.xlsx](./documentation/categories.xlsx).

<img src="./documentation/LLM-based log parsing.png" width="700">

## Licenses

**Note:** Most approaches do not provide a license. Please note that especially for preprint papers, the license might change or, if absent, be added in the future (status 7-March-2025).

| **Approach**                                      | **License**                                      | **Preprint?** |
|---------------------------------------------------|--------------------------------------------------|---------------|
| Cui et al. (LogEval)                              | N/A                                              | ❌             |
| Ji et al. (SuperLog)                              | Apache License Version 2.0, January 2004         | ❌             |
| Jiang et al. (LILAC)                              | N/A                                              |               |
| Liu et al. (LogPrompt)                            | N/A                                              |               |
| Ma et al. (LLMParser)                             | N/A                                              |               |
| Ma et al. (OpenLogParser)                         | N/A                                              | ❌             |
| Mehrabi et al.                                    | N/A                                              |               |
| Pei et al. (SelfLog)                              | N/A                                              |               |
| Sun et al. (Semirald)                             | N/A                                              | ❌             |
| Vaarandi et al. (LLM-TD)                          | GNU General Public License version 2             | ❌             |
| Xiao et al. (LogBatcher)                          | MIT License 2024                                 |               |
| Xu et al. (DivLog)                                | Apache License Version 2.0, January 2004         |               |
| Yu et al. (LogGenius)                             | N/A                                              |               |
| Zhang et al. (Lemur)                              | Apache License Version 2.0, January 2004         | ❌             |

## Setup

Before you can run the parsers and the evaluation you need to execute the setup script:
```
./setup.sh
```

Copy your API keys for OpenAI and TogetherAI into the corresponding files in [keys/](./keys/). CodeLlama was run via [Ollama](https://ollama.com/).

## Code Execution

To run all baseline parsers (non-LLM) on the LogHub-2k datasets please execute:
```
python3 parser_run-no-LLM.py
```

To run all LLM-based parsers on the LogHub-2k datasets please execute:
```
python3 parser_run.py
```

To run all parsers on the LogHub-2.0 datasets please execute:
```
./download.sh
python3 parser_run-full.py
```

In each script you can adjust the used parsers, datasets, LLM, etc.

The results can be found in zip files in the folders [output/](./output/) and [output-full/](./output-full/).

If you do not want to rerun all the parsers you can unzip the output zip files by executing:
```
unzip output.zip
unzip output-full.zip
```
They also contain the result files of the evaluation.

## Evaluation

All plots are given in the notebook file [run_evaluation.ipynb](./run_evaluation.ipynb) and are produced from the csv files within the output folders. For reproduction simply rerun the code within. If you don't want to re-evaluate (it takes some time), simply comment out all the evaluate() functions to reuse the existing results.

To evaluate everything and produce the result files and the plots you can also run:

```
python3 run_evaluation.py
```

## Other

To find the right hyperparameters for the Audit dataset we simply run GridSearch over a selection of parameters. Since this is the baseline we let it run over the entire dataset to get the maximum possible performance:

```
python3 parser_run-no-LLM-GridSearch.py
```