
import subprocess
import time
import json
import os
import signal
import sys
import urllib.request
import urllib.error

SERVER_PORT = 8081
DATA_DIR = "demo_data"
VDB_FILE = os.path.join(DATA_DIR, "demo.vdb")
IDX_FILE = os.path.join(DATA_DIR, "demo.idx")

def setup():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Create a small dummy vectors.json
    # 512 dimensions to match our mock embedding
    vectors = [[0.0] * 512] 
    init_json = os.path.join(DATA_DIR, "init.json")
    with open(init_json, "w") as f:
        json.dump(vectors, f)
        
    print("Building initial .vdb file...")
    subprocess.run([
        "cargo", "run", "--quiet", "--release", "--", "ingest",
        "--input", init_json,
        "--output", VDB_FILE
    ], check=True)

def start_server():
    print(f"Starting server on port {SERVER_PORT}...")
    # Use quiet mode to reduce build noise
    proc = subprocess.Popen([
        "cargo", "run", "--quiet", "--release", "--", "serve",
        "--data", VDB_FILE,
        "--index", IDX_FILE,
        "--port", str(SERVER_PORT),
        "--m", "16",
        "--ef-construction", "100"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc

def wait_for_health(proc):
    print("Waiting for server health...")
    url = f"http://localhost:{SERVER_PORT}/health"
    start = time.time()
    while time.time() - start < 30:
        if proc.poll() is not None:
            print("Server process exited prematurely!")
            print(proc.stderr.read())
            return False
            
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    print("Server is healthy!")
                    print(json.loads(response.read().decode()))
                    return True
        except (urllib.error.URLError, ConnectionResetError):
            time.sleep(1)
            
    print("Timeout waiting for server.")
    return False

def run_demo():
    setup()
    server_proc = start_server()
    
    try:
        if not wait_for_health(server_proc):
            return
            
        # 1. Ingest Audio (Mock)
        print("\n=== Ingesting Mock Audio ===")
        dummy_audio = os.path.join(DATA_DIR, "track1.wav")
        with open(dummy_audio, "w") as f:
            f.write("dummy audio content")
            
        ingest_payload = json.dumps({
            "audio_path": dummy_audio
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"http://localhost:{SERVER_PORT}/ingest",
            data=ingest_payload,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            print(f"Ingest Status: {response.status}")
            print(f"Response: {response.read().decode()}")
            assert response.status == 200 or response.status == 201
        
        # 2. Search
        print("\n=== Searching ===")
        # Search with a random vector (mock embedding size is 512)
        query_vector = [0.1] * 512 
        search_payload = json.dumps({
            "vector": query_vector,
            "k": 5
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"http://localhost:{SERVER_PORT}/search",
            data=search_payload,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            print(f"Search Status: {response.status}")
            result = json.loads(response.read().decode())
            print(f"Response: {json.dumps(result, indent=2)}")
            assert len(result["results"]) > 0
        
        print("\n✅ Demo Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo Failed: {e}")
        # Print server stderr if failed
        if server_proc.poll() is None:
             server_proc.terminate()
        out, err = server_proc.communicate()
        print("Server Stderr:", err)
        
    finally:
        print("Stopping server...")
        if server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait()

if __name__ == "__main__":
    run_demo()
