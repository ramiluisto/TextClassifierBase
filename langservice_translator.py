import pandas as pd
import json
import os

# Assuming you have a DataFrame named df with "text" and "label" columns
df = pd.DataFrame({
    "text": ["example text 1", "example text 2", "example text 3"],
    "label": ["Computer_science", "Psychology", "Biochemistry"]
})

df = pd.read_json('./traintestdata/train.json')

# Define the path where you want to save the text files
output_path = './CogSerFormat'

# Create the directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# The initial part of your JSON structure
data = {
    "projectFileVersion": "2022-05-01",
    "stringIndexType": "Utf16CodeUnit",
    "metadata": {
      "projectName": "WebOfScience",
      "storageInputContainerName": "example-data",
      "projectKind": "CustomSingleLabelClassification",
      "description": "Trying out single label text classification",
      "language": "en",
      "multilingual": False,
      "settings": {}
    },
    "assets": {
      "projectKind": "CustomSingleLabelClassification",
      "classes": [
        # Classes will be added here
      ],
      "documents": [
        # Documents will be added here
      ]
    }
}

# Add classes to the JSON
classes = df['label'].unique()
for class_ in classes:
    data['assets']['classes'].append({"category": class_})

# Save each row's text to a separate .txt file and reference that in the JSON
for i, row in df.iterrows():
    filename = f"{i}.txt"
    with open(os.path.join(output_path, filename), 'w') as f:
        f.write(row['text'])

    # Add document info to the JSON
    data['assets']['documents'].append({
        "location": filename,
        "language": "en-us",
        "class": {
            "category": row['label']
        }
    })

# Save the JSON to a file
with open('CorgSerFormat/labels.json', 'w') as f:
    json.dump(data, f, indent=2)
