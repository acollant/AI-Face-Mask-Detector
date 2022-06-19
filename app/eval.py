import torch
from sklearn.metrics import classification_report

from dataset import dataset, single_test_dl, test_dl, train_dl
from util import construct_confusion_matrix, test_stats

cnn_v3 = torch.load("./model/CNNV3.pb")
cnn_v4 = torch.load("./model/CNNV3_kfold.pb")


predicted_v3, labels_v3, correct_v3, total_v3 = test_stats(cnn_v3, test_dl)
print(
    f"Accuracy of the model on the {total_v3} test images: {round(100 * correct_v3 / total_v3, 2)} %"
)
print(
    f"Classification Report for v3: \n {classification_report(labels_v3, predicted_v3, zero_division=1, target_names=dataset.classes)}"
)
construct_confusion_matrix(labels_v3, predicted_v3, "conf_matrix_v3")

predicted_v4, labels_v4, correct_v4, total_v4 = test_stats(cnn_v4, test_dl)
print(
    f"Accuracy of the model on the {total_v4} test images: {round(100 * correct_v4 / total_v4, 2)} %"
)
print(
    f"Classification Report for v4: \n {classification_report(labels_v4, predicted_v4, zero_division=1, target_names=dataset.classes)}"
)
construct_confusion_matrix(labels_v4, predicted_v4, "conf_matrix_v4")

predicted_single, _, _, _ = test_stats(cnn_v4, single_test_dl)
print(f"The current model predicted: {dataset.classes[predicted_single[0]]}")
