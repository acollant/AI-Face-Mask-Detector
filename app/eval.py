import torch
from sklearn.metrics import classification_report

from dataset import dataset, single_test_dl, test_dl
from util import construct_confusion_matrix, test_stats

cnn_v1 = torch.load("./model/CNNV1.pb")
cnn_v2 = torch.load("./model/CNNV2.pb")
cnn_v3 = torch.load("./model/CNNV3.pb")


predicted_v1, labels_v1, correct_v1, total_v1 = test_stats(cnn_v1, test_dl)
print(
    f"Accuracy of the model on the {total_v1} (v1) test images: {round(100 * correct_v1 / total_v1, 2)} %"
)
print(
    f"Classification Report for v1: \n {classification_report(labels_v1, predicted_v1, zero_division=1, target_names=dataset.classes)}"
)
construct_confusion_matrix(labels_v1, predicted_v1, "conf_matrix_v1")

predicted_v2, labels_v2, correct_v2, total_v2 = test_stats(cnn_v2, test_dl)
print(
    f"Accuracy of the model on the {total_v2} (v2) test images: {round(100 * correct_v2 / total_v2, 2)} %"
)
print(
    f"Classification Report for v2: \n {classification_report(labels_v2, predicted_v2, zero_division=1, target_names=dataset.classes)}"
)
construct_confusion_matrix(labels_v2, predicted_v2, "conf_matrix_v2")

predicted_v3, labels_v3, correct_v3, total_v3 = test_stats(cnn_v3, test_dl)
print(
    f"Accuracy of the model on the {total_v3} test images: {round(100 * correct_v3 / total_v3, 2)} %"
)
print(
    f"Classification Report for v3: \n {classification_report(labels_v3, predicted_v3, zero_division=1, target_names=dataset.classes)}"
)
construct_confusion_matrix(labels_v3, predicted_v3, "conf_matrix_v3")

predicted_single, _, _, _ = test_stats(cnn_v3, single_test_dl)
print(f"The current model predicted: {dataset.classes[predicted_single[0]]}")
