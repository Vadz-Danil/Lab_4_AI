import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/danil/Downloads/data_metrics.csv')

# Додавання прогнозованих міток з порогом 0.5
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')

# Функції для обчислення TP, FN, FP, TN
def find_TP(y_true, y_pred):
    # Обчислення кількості істинно позитивних (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    # Обчислення кількості хибно негативних (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    # Обчислення кількості хибно позитивних (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    # Обчислення кількості істинно негативних (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

# Функція для обчислення всіх значень матриці помилок
def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

# Функція для створення матриці помилок
def vadzinskyi_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# Перевірка матриці помилок
print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

print('Confusion Matrix RF:', vadzinskyi_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
assert np.array_equal(vadzinskyi_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                     confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'vadzinskyi_confusion_matrix() is not correct for RF'
assert np.array_equal(vadzinskyi_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                     confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'vadzinskyi_confusion_matrix() is not correct for LR'

# Функція для обчислення точності (accuracy)
def vadzinskyi_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

assert vadzinskyi_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'vadzinskyi_accuracy_score failed on RF'
assert vadzinskyi_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'vadzinskyi_accuracy_score failed on LR'
print('Accuracy RF: %.3f' % (vadzinskyi_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (vadzinskyi_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

# Функція для обчислення повноти (recall)
def vadzinskyi_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

assert vadzinskyi_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'vadzinskyi_recall_score failed on RF'
assert vadzinskyi_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'vadzinskyi_recall_score failed on LR'
print('Recall RF: %.3f' % (vadzinskyi_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (vadzinskyi_recall_score(df.actual_label.values, df.predicted_LR.values)))

# Функція для обчислення точності (precision)
def vadzinskyi_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

assert vadzinskyi_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'vadzinskyi_precision_score failed on RF'
assert vadzinskyi_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'vadzinskyi_precision_score failed on LR'
print('Precision RF: %.3f' % (vadzinskyi_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (vadzinskyi_precision_score(df.actual_label.values, df.predicted_LR.values)))

# Функція для обчислення F1-міри
def vadzinskyi_f1_score(y_true, y_pred):
    recall = vadzinskyi_recall_score(y_true, y_pred)
    precision = vadzinskyi_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Отладочный вывод для проверки F1-меры
print('Custom F1 LR: %.10f' % vadzinskyi_f1_score(df.actual_label.values, df.predicted_LR.values))
print('Sklearn F1 LR: %.10f' % f1_score(df.actual_label.values, df.predicted_LR.values))

assert np.isclose(vadzinskyi_f1_score(df.actual_label.values, df.predicted_RF.values),
                 f1_score(df.actual_label.values, df.predicted_RF.values), rtol=1e-5), 'vadzinskyi_f1_score failed on RF'
assert np.isclose(vadzinskyi_f1_score(df.actual_label.values, df.predicted_LR.values),
                 f1_score(df.actual_label.values, df.predicted_LR.values), rtol=1e-5), 'vadzinskyi_f1_score failed on LR'
print('F1 RF: %.3f' % (vadzinskyi_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (vadzinskyi_f1_score(df.actual_label.values, df.predicted_LR.values)))

# Порівняння метрик при різних порогах
print('Scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (vadzinskyi_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (vadzinskyi_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (vadzinskyi_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (vadzinskyi_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('Scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (vadzinskyi_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (vadzinskyi_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (vadzinskyi_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (vadzinskyi_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

# Побудова ROC-кривих
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

# Обчислення AUC
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

# Побудова графіку ROC-кривих
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()