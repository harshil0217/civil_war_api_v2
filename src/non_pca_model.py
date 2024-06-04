import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shap
import pickle
from joblib import dump


#load the preprocessed data
X_train, X_test, y_train, y_test = preprocess_data(pca=False)

#creating sequential model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dropout(0.25))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

#creating optimizer
adam = keras.optimizers.Adam(learning_rate=0.001)

#compile the model
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

#train the model
model.fit(X_train, y_train, epochs=100)

#evaluate the model
loss_and_metrics = model.evaluate(X_test, y_test)
print(loss_and_metrics)
print('Loss = ',loss_and_metrics[0])
print('Accuracy = ',loss_and_metrics[1])

#see predictions
preds = model.predict(X_test)
preds = np.round(preds)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./images/confusion_matrix_non_pca.png')
print(classification_report(y_test, preds))

# Calculate ROC curve
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Civil War Prediction')
plt.legend()
plt.savefig('./images/roc_curve_non_pca.png')

# Save the model
model.save('./models/civil_war_model_non_pca.h5')

# Save the training data
X_train = pd.DataFrame(X_train)
X_train.to_csv('./data/X_train_non_pca.csv', index=False)