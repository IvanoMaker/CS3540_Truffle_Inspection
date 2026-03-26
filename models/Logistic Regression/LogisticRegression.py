import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from past.builtins import xrange
from scipy.special import expit

defective = "C:/Users/boyle/PycharmProjects/TruffleDefectDetection/dataset/Defective"
satisfactory = "C:/Users/boyle/PycharmProjects/TruffleDefectDetection/dataset/Satisfactory"

def FlattenImages(folder):
    data = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        flattened_img = img.flatten()
        data.append(flattened_img)


