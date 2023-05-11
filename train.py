from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import os


eval_df = pd.read_json('./traintestdata/test.json')
train_df = pd.read_json('./traintestdata/train.json')
full_df = pd.read_json('./traintestdata/full_data.json')

model_args = ClassificationArgs(
    use_multiprocessing=False,
    use_multiprocessing_for_evaluation=False,
    num_train_epochs=8
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


label_names = list(full_df.label.unique())
label_names.sort()
code_to_label = dict(enumerate(label_names))
label_to_code = {label : code for code, label in code_to_label.items()}

train_df['label'] = train_df.label.replace(label_to_code)
eval_df['label'] = eval_df.label.replace(label_to_code)

label_count = len(label_names)
print(80*'*')
print(f"I see {label_count} labels:")
print(label_to_code)
print(train_df.shape)
print(train_df.head())
# Create a ClassificationModel
model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=label_count, args = model_args)#, args={'learning_rate':1e-5, 'num_train_epochs': 4, 'reprocess_input_data': True, 'overwrite_output_dir': True})



model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')
    
result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_multiclass, acc=accuracy_score)

print(result)