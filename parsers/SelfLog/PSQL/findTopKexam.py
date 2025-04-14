from sshtunnel import SSHTunnelForwarder
from PSQL.conConfig import ssh_config, postgresql_config
import psycopg2
from PSQL.conConfig import model



def findTopK(log,model):
    result = []
    with SSHTunnelForwarder(**ssh_config) as tunnel:
        conn = psycopg2.connect(
            host='127.0.0.1',
            port=tunnel.local_bind_port,
            **postgresql_config
        )

        vector_val = model.encode(log, normalize_embeddings=True)
        query = '''
            SELECT log,template, 1 - ("logVector" <=> %s::VECTOR) AS cosine_similarity
            FROM log_template
            ORDER BY cosine_similarity DESC
            LIMIT 3;
        '''
        with conn.cursor() as cursor:
            cursor.execute(query, (list(vector_val),))
            result = cursor.fetchall()
        for row in result:
            print(f"log: {row[0]}, Template: {row[1]}, similarity: {row[2]}")

        conn.close()
    return result