import os

import pandas as pd
import psycopg2
from sshtunnel import SSHTunnelForwarder
from tqdm import tqdm

from conConfig import *

directory_path = "./cand_export/"

seed = 42

def random_sample_from_folders(directory, fraction, columns_of_interest):
    folder_paths = [os.path.join(directory, folder) for folder in os.listdir(directory) if
                    os.path.isdir(os.path.join(directory, folder))]

    result_list = []
    for folder_path in folder_paths:
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('log_templates.csv')]
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, usecols=columns_of_interest)
            result_list.append(df.sample(frac=fraction, random_state=seed).to_numpy().tolist())

    return result_list

#with SSHTunnelForwarder(**ssh_config) as tunnel:
conn = psycopg2.connect(
    host='127.0.0.1',
    port="5432", #tunnel.local_bind_port,
    **postgresql_config
)

cur = conn.cursor()

selected_fraction = 1
selected_columns = ['Content', 'EventTemplate']
result = random_sample_from_folders(directory_path, selected_fraction, selected_columns)
for system_selectlog in tqdm(result):
    for log,template in tqdm(system_selectlog):
        log_val = log
        template_val = template
        vector_val = model.encode(log_val, normalize_embeddings=True)

        cur.execute("""INSERT INTO public.log_template (
                        "ID", log, template, "logVector")
                        VALUES (nextval('id_seq'), %s, %s, %s)""",
                    (log_val, template_val, list(vector_val)))
        conn.commit()
cur.close()
conn.close()
