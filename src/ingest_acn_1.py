import os
import json
import uuid
from datetime import datetime
from collections import defaultdict
from acnportal import acndata
from hdfs import InsecureClient

# --- CONFIG ---
HDFS_URL = "http://192.168.40.130:9870"
HDFS_USER = "axuan"
BASE_PATH = "/ev-project/data/bronze/ev_sessions/caltech"
API_KEY = os.getenv("ACN_API_KEY")
BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- INIT ---
api_client = acndata.DataClient(API_KEY)
hdfs_client = InsecureClient(HDFS_URL, user=HDFS_USER)

def run_ingestion():
    # 1. Fetch
    sessions = api_client.get_sessions(site="caltech", sort="-connectionTime")
    
    # 2. Partitioning & Metadata
    partitioned_data = defaultdict(list)
    for s in sessions:
        # Normalize time
        ts = s["connectionTime"]
        dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
        
        # Define partition path
        p_path = dt.strftime("year=%Y/month=%m/day=%d")
        
        # Add Metadata
        s["_ingest_time"] = datetime.utcnow().isoformat()
        s["_batch_id"] = BATCH_ID
        
        partitioned_data[p_path].append(s)

    # 3. Write to Data Lake
    for p_path, records in partitioned_data.items():
        full_path = f"{BASE_PATH}/{p_path}"
        hdfs_client.makedirs(full_path)
        
        # File name is unique for Idempotency
        file_name = f"part_{BATCH_ID}_{uuid.uuid4().hex[:8]}.json"
        
        # Convert to JSON Lines
        content = "\n".join(json.dumps(r, default=str) for r in records)
        
        with hdfs_client.write(f"{full_path}/{file_name}", encoding="utf-8") as writer:
            writer.write(content)
            
    print(f"Ingestion {BATCH_ID} successful.")

if __name__ == "__main__":
    run_ingestion()