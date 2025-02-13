"""
This module provides functions to set up and load a PostgreSQL database with a PGVectorStore.
Functions:
- setup_postgresql_database(user, password, host, port, db_name, hnsw_kwargs=None, embed_dim=1024):
- load_postgresql_database(user, password, host, port, db_name):
"""
import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore

def setup_postgresql_database(user,
                              password,
                              host,
                              port,
                              db_name,
                              hnsw_kwargs=None,
                              embed_dim=1024,
                              hybrid=False
    ):
    """
    Set up a PostgreSQL database and initialize a PGVectorStore.

    Parameters:
    user (str): Database user.
    password (str): Database password.
    host (str): Database host.
    port (int): Database port.
    db_name (str): Name of the database to create.
    table_name (str): Name of the table to create.
    embed_dim (int): Embedding dimension for the PGVectorStore.
    hnsw_kwargs (dict): HNSW parameters for the PGVectorStore.
    hybrid (bool): Hybrid search flag for the PGVectorStore.

    Returns:
    PGVectorStore: Initialized PGVectorStore object.
    """

    # PostgreSQL connection setup
    connection_string = f"user={user} password={password} host={host} port={port}"

    conn = psycopg2.connect(connection_string)
    conn.autocommit = True
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as c:
        # Terminate active connections to the target database
        c.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s
            AND pid <> pg_backend_pid();
        """, (db_name,))
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")

    # Set default hnsw_kwargs if not given
    if hnsw_kwargs is None:
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        }

    pg_vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="tourism_vectors",
        embed_dim=embed_dim,
        hnsw_kwargs=hnsw_kwargs,
        hybrid_search=hybrid
    )
    conn.close()
    return pg_vector_store

def load_postgresql_database(user, password, host, port, db_name):
    """
    Load a PostgreSQL database and return a PGVectorStore.

    Parameters:
    user (str): Database user.
    password (str): Database password.
    host (str): Database host.
    port (int): Database port.
    db_name (str): Name of the database to load.
    table_name (str): Name of the table to load.
    
    Returns:
    PGVectorStore: Loaded PGVectorStore object.
    """

    pg_vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="tourism_vectors",
    )

    return pg_vector_store
