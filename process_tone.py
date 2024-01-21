import json
import sys

def combine_json_objects(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        combined_dict = {}
        for item in data:
            for key, value in item.items():
                if key in combined_dict:
                    for subkey, subvalue in value.items():
                        if subkey in combined_dict[key]:
                            combined_dict[key][subkey].extend(subvalue)
                        else:
                            combined_dict[key][subkey] = subvalue
                else:
                    combined_dict[key] = value

        return combined_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# Example usage

file_path = sys.argv[1]
combined_data = combine_json_objects(file_path)
print(json.dumps(combined_data, indent=4))