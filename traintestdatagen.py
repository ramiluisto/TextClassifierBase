import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_FOLDER = "./raw_data/"
TARGET_DATA_FOLDER = "./traintestdata"


def main():

    classes = {}
    for filename in os.listdir(RAW_DATA_FOLDER):
        datum = extract_data_from_file(filename)
        classes[filename] = datum

    full_df = create_full_dataset(classes)

    print('\n\n')
    print(full_df.shape)
    print(full_df.label.value_counts())

    train_df, test_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["label"]
    )

    full_df.to_json(os.path.join(TARGET_DATA_FOLDER, "full_data.json"))
    train_df.to_json(os.path.join(TARGET_DATA_FOLDER, "train.json"))
    test_df.to_json(os.path.join(TARGET_DATA_FOLDER, "test.json"))


def extract_data_from_file(filename):
    filepath = os.path.join(RAW_DATA_FOLDER, filename)
    with open(filepath, "r") as fp:
        lines = fp.readlines()

    total_lines = len(lines)
    start_line = int(total_lines * 0.1)
    stop_line = int(total_lines * 0.9)
    cropped_lines = lines[start_line:stop_line]

    #cropped_text = ''.join(cropped_lines)
    #lines = cropped_text.split('\n\n')

    merged_lines = []

    previous_line = ''
    for line in cropped_lines:
        if len(previous_line) < 800:
            previous_line += '\n\n'
            previous_line += line
        else:
            merged_lines.append(previous_line)
            previous_line = ''

    if previous_line != '':
        merged_lines.append(previous_line)        

    print(f"Found {len(merged_lines):6} lines from {filename}.")
    return merged_lines    

def create_full_dataset(classes):
    texts = []
    labels = []
    for name, text_data in classes.items():
        label = os.path.splitext(name)[0]
        for text in text_data:
            texts.append(text)
            labels.append(label)

    data = {
        'text' : texts,
        'label' : labels
    }
    df = pd.DataFrame.from_dict(data)

    return df


if __name__ == "__main__":
    main()
