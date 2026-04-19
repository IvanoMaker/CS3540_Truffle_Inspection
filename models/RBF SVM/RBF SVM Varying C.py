from xml.parsers.expat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from sklearn.svm import SVC
import joblib



# ============================== Load Data ==============================
x_npy = np.load('x_data_flattened.npy')
y_npy = np.load('y_labels.npy')

# ============================== Split into Training and Testing ==============================
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size=0.25, random_state=0)
print("Training x:", x_train.shape,"y:", y_train.shape)
print("Testing x:", x_test.shape,"y:", y_test.shape)

# balance the dataset by removing some of the good samples until the number of good and defective samples are equal in the training set
defectCount = np.sum(y_train == 1)

# print("Number of defective samples in training set:", defectCount)

# goodCount = 0
# for i, y in enumerate(y_train):
#     if goodCount <= defectCount:
#         if y == 0:
#             goodCount += 1
#     else:
#         if y == 0:
#             y_train[i] = -1  # Mark for removal
# # Remove marked samples
# x_train = x_train[y_train != -1]
# y_train = y_train[y_train != -1]

# print("Balanced Training x:", x_train.shape,"y:", y_train.shape)
# print("Number of good samples in training set:", np.sum(y_train == 0))

# # balance the test set by removing some of the good samples until the number of good and defective samples are equal in the test set
# defectCount = np.sum(y_test == 1)
# print("Number of defective samples in test set:", defectCount)
# goodCount = 0
# for i, y in enumerate(y_test):
#     if goodCount <= defectCount:
#         if y == 0:
#             goodCount += 1
#     else:
#         if y == 0:
#             y_test[i] = -1  # Mark for removal
# # Remove marked samples
# x_test = x_test[y_test != -1]
# y_test = y_test[y_test != -1]
# print("Balanced Testing x:", x_test.shape,"y:", y_test.shape)
# print("Number of good samples in test set:", np.sum(y_test == 0))

accuracy = np.zeros(5)
c = [0.01, 0.1, 1, 10, 100]

for i, C in enumerate(c):
    print("C:", C)
    # ============================== Train Model ==============================
    svc = SVC(kernel='rbf', C=C, probability=True)
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    accuracy[i] = svc.score(x_test, y_test)
    joblib.dump(svc, f'SVC_model_C={C}.joblib')

    # ============================== Show Confusion Matrix ==============================
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    plt.imshow(conf_matrix)
    plt.ylabel("Actual Value")
    plt.xlabel("Predicted Value")
    plt.colorbar()
    plt.savefig(f'confusion_matrix_C={C}.png')
    plt.clf()

    # ============================== Create ROC ==============================
    RocCurveDisplay.from_estimator(svc, x_test, y_test)
    plt.title(f'ROC Curve for C={C}')
    plt.savefig(f'ROC_C={C}.png')
    plt.clf()

    # ============================== Show Statistics ==============================
    class_report = classification_report(y_test, predictions)
    with open(f'classification_report_C={C}.txt', 'w') as file:
        file.write(class_report)
    print("====================================================================================================")

# ============================== Save Best Model ==============================
best_c = c[np.argmax(accuracy)]
print(f"\nBest C value: {best_c} with accuracy: {np.max(accuracy):.4f}")
best_model = joblib.load(f'SVC_model_C={best_c}.joblib')
joblib.dump(best_model, 'SVM_model.joblib')
print(f"Best model saved as SVM_model.joblib")

# ============================== Plot Accuracy vs C ==============================
plt.plot(c, accuracy)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.title('Accuracy vs. C')
plt.savefig('Accuracy vs C.png')