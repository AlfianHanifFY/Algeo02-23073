import json
from API.loadJson import load_json

def updateConfig(param, value):
    config = load_json("config/config.json")
    print(config)
    data = config
    data[0][param] = value
    # Save as a JSON file
    output_file = "test/config/config.json"
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON data has been saved to {output_file}")
