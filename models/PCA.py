import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

x_npy = np.load('x_data_flattened.npy')
y_npy = np.load('y_labels.npy')

df = pd.DataFrame(x_npy)
df['target'] = y_npy
COMPONENTS = 2

pca = PCA(n_components=COMPONENTS)
transformed_data = pca.fit_transform(df.iloc[:,:-1].to_numpy())

mask_satisfactory = (y_npy != 1)
mask_defective = (y_npy == 1)
plt.scatter(transformed_data[mask_satisfactory, 0], transformed_data[mask_satisfactory, 1], c='green', alpha=0.5, label='satisfactory')
plt.scatter(transformed_data[mask_defective, 0], transformed_data[mask_defective, 1], c='red', alpha=0.5, label='defective')
plt.suptitle("Primary Components Projection Map")
plt.title(f"components={COMPONENTS}")
plt.legend()
plt.show()