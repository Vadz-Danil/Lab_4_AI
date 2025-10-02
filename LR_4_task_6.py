import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from utilities import visualize_classifier


# Завантаження даних
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Наївний Байєс
nb_classifier = GaussianNB()
nb_classifier.fit(X, y)
y_pred_nb = nb_classifier.predict(X)
nb_accuracy_train = 100.0 * accuracy_score(y, y_pred_nb)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Розмір тестового набору: {X_test.shape}")
nb_classifier_new = GaussianNB()
nb_classifier_new.fit(X_train, y_train)
y_test_pred_nb = nb_classifier_new.predict(X_test)
nb_accuracy_test = 100.0 * accuracy_score(y_test, y_test_pred_nb)

num_folds = 3
nb_accuracy_cv = cross_val_score(nb_classifier, X, y, scoring='accuracy', cv=num_folds).mean() * 100
nb_precision_cv = cross_val_score(nb_classifier, X, y, scoring='precision_weighted', cv=num_folds).mean() * 100
nb_recall_cv = cross_val_score(nb_classifier, X, y, scoring='recall_weighted', cv=num_folds).mean() * 100
nb_f1_cv = cross_val_score(nb_classifier, X, y, scoring='f1_weighted', cv=num_folds).mean() * 100

print("\n=== Результати Наївного Байєса (GaussianNB) ===")
print(f"Accuracy на тренувальних даних: {nb_accuracy_train:.2f}%")
print(f"Accuracy на тестовому наборі: {nb_accuracy_test:.2f}%")
print(f"Перехресна перевірка (3 фолди):")
print(f"  Accuracy: {nb_accuracy_cv:.2f}%")
print(f"  Precision: {nb_precision_cv:.2f}%")
print(f"  Recall: {nb_recall_cv:.2f}%")
print(f"  F1-score: {nb_f1_cv:.2f}%")

visualize_classifier(nb_classifier_new, X_test, y_test)
plt.title("Візуалізація Наївного Байєса на тестовому наборі")
plt.show()
plt.close()

# SVM
svm_classifier = SVC(kernel='linear', C=1, random_state=3)
svm_classifier.fit(X, y)
y_pred_svm = svm_classifier.predict(X)
svm_accuracy_train = 100.0 * accuracy_score(y, y_pred_svm)

svm_classifier_new = SVC(kernel='linear', C=1, random_state=3)
svm_classifier_new.fit(X_train, y_train)
y_test_pred_svm = svm_classifier_new.predict(X_test)
svm_accuracy_test = 100.0 * accuracy_score(y_test, y_test_pred_svm)

svm_accuracy_cv = cross_val_score(svm_classifier, X, y, scoring='accuracy', cv=num_folds).mean() * 100
svm_precision_cv = cross_val_score(svm_classifier, X, y, scoring='precision_weighted', cv=num_folds).mean() * 100
svm_recall_cv = cross_val_score(svm_classifier, X, y, scoring='recall_weighted', cv=num_folds).mean() * 100
svm_f1_cv = cross_val_score(svm_classifier, X, y, scoring='f1_weighted', cv=num_folds).mean() * 100

print("\n=== Результати SVM (лінійне ядро) ===")
print(f"Accuracy на тренувальних даних: {svm_accuracy_train:.2f}%")
print(f"Accuracy на тестовому наборі: {svm_accuracy_test:.2f}%")
print(f"Перехресна перевірка (3 фолди):")
print(f"  Accuracy: {svm_accuracy_cv:.2f}%")
print(f"  Precision: {svm_precision_cv:.2f}%")
print(f"  Recall: {svm_recall_cv:.2f}%")
print(f"  F1-score: {svm_f1_cv:.2f}%")

visualize_classifier(svm_classifier_new, X_test, y_test)
plt.title("Візуалізація SVM на тестовому наборі")
plt.show()
plt.close()
