import subprocess
import json
import os 

# BASE_DIR = os.path.dirname(__file__)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "./"

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    print(f"Output:\n{err.decode('utf-8')}")
    print(f"Output:\n{out.decode('utf-8')}")
    process.kill()
    return out, err


def get_tools() -> list[dict]:
    return json.load(open(os.path.join(BASE_DIR, 'app/fn_tools.json')))