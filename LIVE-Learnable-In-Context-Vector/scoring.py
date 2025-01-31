import json
from evaluate import load


result_path = "./result/inference/idefics2-8b-base/vqav2/vqav2_idefics_icv/meta_info/icl_shot1.json"
# result_path = "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/result/inference/idefics2-8b-base/vqav2/vqav2_idefics_icv/meta_info/2025-01-11 09:45:33.816168-cfg.test_num=-1-icv.json"

result_path = "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/result/inference/idefics2-8b-base/vqav2/fhm_idefics_icv/meta_info/icl_shot1.json"

with open(result_path, "r") as content:
    result_file = json.load(content)

references = []
predictions = []

for idx, content in result_file.items():
    references.append(content["answer"])
    predictions.append(content["prediction"].split(".")[0])

# print(refereces)
# bertscore = load("bertscore")

# results = bertscore.compute(predictions=predictions, references=references, lang="en")


# f1_scores = results["f1"]
# print(f"BERTScore (F1) = {sum(f1_scores)/len(f1_scores)}")

# with open("./data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json", "r") as content:
#     train  = json.load(content)

# print(len(train["questions"]))

from sklearn.metrics import f1_score, roc_auc_score
import ast

# Define ground truth and predictions
# ground_truth = ['[\'pc_empty\']', '[\'sex\', \'religion\']']
# predictions = ['[\'race\']', '[\'sex\', \'religion\']']

# Convert strings to lists
ground_truth = [ast.literal_eval(gt) for gt in references]
predictions = [ast.literal_eval(pred) for pred in predictions]

# print(ground_truth)
# print(predictions)

# Create a set of all unique categories for multi-label binarization
categories = sorted({cat for sublist in ground_truth + predictions for cat in sublist})

print(categories)

# Convert lists to binary format for each category
ground_truth_bin = [[1 if category in gt else 0 for category in categories] for gt in ground_truth]
predictions_bin = [[1 if category in pred else 0 for category in categories] for pred in predictions]

# Calculate F1 scores
f1_micro = f1_score(ground_truth_bin, predictions_bin, average='micro')
f1_macro = f1_score(ground_truth_bin, predictions_bin, average='macro')

print(f1_micro, f1_macro)

# Calculate ROC-AUC scores
roc_auc_micro = roc_auc_score(ground_truth_bin, predictions_bin, average='micro', multi_class='ovr')
roc_auc_macro = roc_auc_score(ground_truth_bin, predictions_bin, average='macro', multi_class='ovr')

print(roc_auc_micro, roc_auc_macro)

