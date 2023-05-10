from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd



eval_df = pd.read_json('./traintestdata/test.json')
train_df = pd.read_json('./traintestdata/train.json')
full_df = pd.read_json('./traintestdata/full_data.json')


# Create a ClassificationModel
model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=len(full_df.label.unique()), args={'learning_rate':1e-5, 'num_train_epochs': 4, 'reprocess_input_data': True, 'overwrite_output_dir': True}, use_cuda=False)


model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')
    
result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_multiclass, acc=accuracy_score)

print(result)