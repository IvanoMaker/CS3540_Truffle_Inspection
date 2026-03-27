import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# ============================== Load Data ==============================
x_npy = np.load('../../x_data_flattened.npy')
y_npy = np.load('../../y_labels.npy')

# ============================== Split into Training and Testing ==============================
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size=0.25, random_state=0)
print("Training x:", x_train.shape,"y:", y_train.shape)
print("Testing x:", x_test.shape,"y:", y_test.shape)

cost = np.zeros(10)

for degree in range(1, 11):
    print("Degree:", degree)
    # ============================== Train Model ==============================
    svc = SVC(kernel='poly', degree = degree)
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    cost[degree-1] = svc.score(x_test, y_test)

    # ============================== Show Confusion Matrix ==============================
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    plt.imshow(conf_matrix)
    plt.ylabel("Actual Value")
    plt.xlabel("Predicted Value")
    plt.colorbar()
    plt.savefig(f'confusion_matrix_deg={degree}.png')
    plt.clf()

    # ============================== Show Statistics ==============================
    class_report = classification_report(y_test, predictions)
    print(class_report)
    print("====================================================================================================")

plt.plot(range(1,11), cost)
plt.ylabel('Accuracy')
plt.xlabel('Degree')
plt.title('Accuracy vs. Degree')
plt.savefig('Accuracy vs Degree.png')