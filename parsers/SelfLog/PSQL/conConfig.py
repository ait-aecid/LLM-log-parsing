from psycopg2.extensions import register_adapter, AsIs
import numpy as np
from sentence_transformers import SentenceTransformer

ssh_config = {
    'ssh_address_or_host': ('ip', 1),
    'ssh_username': 'user',
    'ssh_password': 'passwd',
    'remote_bind_address': ('localhost', 5432),
}

postgresql_config = {
    'database': '',
    'user': '',
    'password': '',
}

model_path = ""
register_adapter(np.float32, lambda x: AsIs(x))

model = SentenceTransformer(model_path)