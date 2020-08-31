import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('E:\\myPractice\\src\\main\\resources\\dataset.csv', header="1")
cols = ["WindNumber", "Time", "WindSpeed", "Power", "RotorSpeed"]
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.2)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f ', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()
plt.savefig('sale_corr.png')