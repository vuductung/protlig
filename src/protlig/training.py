import json

def load_json(path):
    with open(path) as fh:
        json.load(fh)
        print("Successfully loaded json file.")