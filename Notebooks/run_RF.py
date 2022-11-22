import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import sklearn
from sklearn.model_selection import train_test_split
import pickle

import sys 

run_id = sys.argv[1]  # Get run ID
africa = pd.read_csv('../Data/africa-ML-54_21-03-11.csv')
africa = africa.dropna('columns')
africa=sklearn.utils.shuffle(africa)
labels = africa['sahoStatusEHA']
data = africa.drop(columns=['sahoStatusEHA', 'country', 
                        #'christCountry', 'cumCasesLag1',
                        #'cumCasesLag1P100KC', 'cumDeathsLag1', 'ebolaCasesN', 'gdpPerCap', 'medDocsN',
                        'tempDiffS', 'tempDiff66', 'dayNum']
                   ).select_dtypes(['number'])
#Split into test and training
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.24, random_state=42, stratify=labels)




feature_importance = {}
for i in np.arange(100):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=5, criterion='log_loss')
    rf.fit(X_train, y_train)
    # Collect most important features
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(rf.feature_importances_, 3)})
    importance.sort_values('importance', ascending=True, inplace=True)
    importance = importance#[:50]
    # Assign values to dictionary
    for imp in importance.values:
        if imp[0] not in feature_importance.keys():
            # If the key is not already there then add it
            feature_importance[imp[0]] = [imp[1]]
        else:
            feature_importance[imp[0]].append(imp[1])
    rf = None

feature_importance_df = pd.DataFrame.from_dict(feature_importance, orient='index')
final_features = feature_importance_df.agg(['mean', 'std', 'sem'], axis=1).sort_values('mean', ascending=False)#[:30]
final_features['95-conf'] = 1.96*final_features['sem']
final_features['99-conf'] = 3*final_features['sem']
pickle.dump(final_features, open('FinalFeatures/final_features_%s.pkl'%run_id,'wb'))
print(final_features[:10])