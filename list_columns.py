import pyarrow.parquet as pq
schema = pq.read_schema("data/hmda_all_clean.parquet")
print(schema.names)