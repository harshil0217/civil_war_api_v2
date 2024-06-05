from data_preprocessing import preprocess_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier


#load the preprocessed data
X_train, X_test, y_train, y_test = preprocess_data(pca=False)

#train XGBoost model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

#evaluate xgb model
xgb_preds = xgb.predict(X_test)
cm = confusion_matrix(y_test, xgb_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./images/confusion_matrix_xgb.png')
print(classification_report(y_test, xgb_preds))

# Calculate ROC curve
y_pred = xgb.predict(X_test).ravel()
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
plt.savefig('./images/roc_curve_xgb.png')

#save model
xgb.save_model('./models/xgb_model.json')


