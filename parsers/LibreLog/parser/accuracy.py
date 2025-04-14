import pandas as pd
import numpy as np

def evaluate_result(predic_file,groundtruth):
    df_gtlog = pd.read_csv(
        groundtruth,  usecols=["Content", "EventId", "EventTemplate"]
    )
    print("df_gtlog file loaded! ", flush=True)
    df_gtlog["EventTemplate_NoSpaces"] = df_gtlog["EventTemplate"].str.replace('\s+', '', regex=True).str.replace(r'\<\*\>', '', regex=True)
    print("df_gtlog EventTemplate ready to be checked", flush=True)
    column_names = ['Content',  'EventTemplate']
    df_parsedlog = pd.read_csv(predic_file, index_col=False,  usecols=column_names)
    print("df_parsedlog file loaded! ", flush=True)
    df_parsedlog["Predict_NoSpaces"] = df_parsedlog['EventTemplate'].str.replace('\s+', '', regex=True).str.replace(r'\(\.\*\?\)', '', regex=True)
    print("df_parsedlog ready to be checked! ", flush=True)
    # df_parsedlog["EventTemplate_NoSpaces"] = df_parsedlog['EventTemplate'].str.replace('\s+', '', regex=True)
    correctly_parsed_messages = df_parsedlog['Predict_NoSpaces'].eq(df_gtlog['EventTemplate_NoSpaces']).values.sum()
    PA = float(correctly_parsed_messages) / len(df_parsedlog[['Content']])
    print(f"PA: {PA}", flush=True)
    # print(f"PA: {PA}", flush=True)
    (precision, recall, f_measure, GA) = get_accuracy(df_gtlog["EventTemplate_NoSpaces"],
                                                               df_parsedlog['Predict_NoSpaces'])
    print(f"accuracy_GA: {GA}", flush=True)
    event_count = str(df_parsedlog["Predict_NoSpaces"].nunique())
    return GA, PA,event_count

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                    logIds.size
                    == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
    precision = 0
    recall = 0
    f_measure = 0
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

