import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from sklearn.svm import SVC

# ============================== Load Data ==============================
x_npy = np.load('../../x_data_flattened.npy')
y_npy = np.load('../../y_labels.npy')

# ============================== Split into Training and Testing ==============================
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size=0.25, random_state=0)
print("Training x:", x_train.shape,"y:", y_train.shape)
print("Testing x:", x_test.shape,"y:", y_test.shape)

accuracy = np.zeros(7)
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i,C in enumerate(c):
    print("C:", C)
    # ============================== Train Model ==============================
    svc = SVC(kernel='rbf', C = C)
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    accuracy[i] = svc.score(x_test, y_test)

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

plt.plot(c, accuracy)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.title('Accuracy vs. C')
plt.savefig('Accuracy vs C.png')