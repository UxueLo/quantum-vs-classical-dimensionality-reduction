import numpy as np
import umap
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, fbeta_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
#Functions that apply classic ML methods

#def classification(X_train, y_train, X_test, y_test):
#def regression(X_train, y_train, X_test, y_test):
#def clustering(X_train, y_train, X_test, y_test):

#Function that receives a dataset and applies the dimensionality reduction methods 
def reduce_all(X_train, y_train, X_test, y_test):
    
    # Number of dimensions we want after the reduction
    n_qubits = 4

    # KernelPCA. It's fit only on train
    kpca = KernelPCA(n_components=n_qubits, kernel="rbf")
    X_train_pca = kpca.fit_transform(X_train)
    X_test_pca = kpca.transform(X_test)

    # UMAP, the random_state hyperparameter makes the result reproducible between runs
    reducer = umap.UMAP(n_components=n_qubits, random_state=42)
    X_train_umap = reducer.fit_transform(X_train)
    X_test_umap = reducer.transform(X_test)

    #Scale the reduced features to a valid angle range. It's fit only on train
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    #Create a helper that runs a circuit on one sample and returns one real value per qubit
    def expvals(circuit, x, pauli):
        state = Statevector(circuit.assign_parameters(x))
        return [
            state.expectation_value(
                SparsePauliOp.from_sparse_list([(pauli, [i], 1)], num_qubits=n_qubits)
            ).real
            for i in range(n_qubits)
        ]

    # Angle Encoding
    params = ParameterVector("x", n_qubits)
    angle_circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        angle_circuit.ry(params[i], i)

    X_train_ae = np.array([expvals(angle_circuit, x, "Z") for x in X_train_scaled])
    X_test_ae = np.array([expvals(angle_circuit, x, "Z") for x in X_test_scaled])

    # ZZFeatureMap
    zz_circuit = ZZFeatureMap(n_qubits)

    X_train_z_map = np.array([expvals(zz_circuit, x, "X") for x in X_train_scaled])
    X_test_z_map = np.array([expvals(zz_circuit, x, "X") for x in X_test_scaled])

    # Return every reduced dataset, named with it's reduction method
    return {
        "PCA": (X_train_pca, y_train, X_test_pca, y_test),
        "UMAP": (X_train_umap, y_train, X_test_umap, y_test),
        "AE": (X_train_ae, y_train, X_test_ae, y_test),
        "Z_MAP": (X_train_z_map, y_train, X_test_z_map, y_test),
    }

#Function that computes all the evaluation metrics seen in class (03_c_Model_Evaluation)
def evaluation(y_test, y_pred, y_proba, classes):
    # Confusion matrix: row i = real class i, column j = predicted class j
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # TP, FP, FN and TN of every class (one class against the rest)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    # Sensitivity (TP/P) and specificity (TN/N)
    if len(classes) == 2:
        # Binary problem: of the positive class (the second label)
        sensitivity = tp[1] / (tp[1] + fn[1])
        specificity = tn[1] / (tn[1] + fp[1])
    else:
        # Multiclass problem: of each class against the rest, then the macro average
        sensitivity = np.mean(tp / (tp + fn))
        specificity = np.mean(tn / (tn + fp))

    # ROC AUC and PR AUC need the probability of each class
    if len(classes) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        pr_auc = average_precision_score(y_test, y_proba[:, 1], pos_label=classes[1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes)
        y_bin = label_binarize(y_test, classes=classes)
        pr_auc = average_precision_score(y_bin, y_proba, average="macro")

    # Macro average: every class has the same weight (recommended in class when no class is more important)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "error_rate": 1 - accuracy_score(y_test, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "g_mean": np.sqrt(sensitivity * specificity),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f2": fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }
#Functions that receive a dataset and prepare the sets for training and testing before applying a traditional ML technique
#Fucntion that applies classification
def classification(X_train, y_train, X_test, y_test):
    # Same cross validation as in the Naive Bayes notebook
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Hyperparameter configurations we want to try for each model
    configs = []

    # CART (Decision Tree)
    #  - splitting criterion: Gini index (CART) or entropy 
    for criterion in ["gini", "entropy"]:
        for max_depth in [2, 3, 5, None]:
            for min_samples_leaf in [1, 5]:
                for ccp_alpha in [0.0, 0.01, 0.05]:
                    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 ccp_alpha=ccp_alpha, random_state=42)
                    hyperparameters = {"splitting_criterion": criterion,
                                       "max_depth (prepruning)": max_depth,
                                       "min_samples_leaf (prepruning)": min_samples_leaf,
                                       "cost_complexity_alpha (postpruning)": ccp_alpha}
                    configs.append(("CART", clf, hyperparameters))

    # Gaussian Naive Bayes
    #  - var_smoothing: the small positive variance floor used when a feature
    #    has zero variance inside a class
    for var_smoothing in [1e-12, 1e-9, 1e-6, 1e-3]:
        clf = GaussianNB(var_smoothing=var_smoothing)
        hyperparameters = {"variance_floor (var_smoothing)": var_smoothing}
        configs.append(("Gaussian Naive Bayes", clf, hyperparameters))

    results = []
    for name, clf, hyperparameters in configs:
        # Average accuracy of the 5 folds, using only the train set
        accs = cross_val_score(clf, X_train, y_train, cv=kf)
        cv_acc = sum(accs) / len(accs)

        # Train with the whole train set and evaluate on test
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        metrics = evaluation(y_test, y_pred, y_proba, clf.classes_)

        results.append({"model": name, "hyperparameters": hyperparameters,
                        "cv_accuracy": round(float(cv_acc), 3),
                        "test_metrics": {k: (round(float(v), 3) if k != "confusion_matrix" else v)
                                         for k, v in metrics.items()}})

    # Best hyperparameter configuration of each model according to the cross validation
    best = {}
    for r in results:
        if r["model"] not in best or r["cv_accuracy"] > best[r["model"]]["cv_accuracy"]:
            best[r["model"]] = r
    return list(best.values())
def data_class(df):
    # For every dataset the classification target feature will be index = 0
    y = df[:, 0]
    X = df[:, 1:]

    # We split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reduced = reduce_all(X_train, y_train, X_test, y_test)

    for name, (Xtr, ytr, Xte, yte) in reduced.items():
        print(name)
        for r in classification(Xtr, ytr, Xte, yte):
            print("  ", r)

#Regression application
def data_regr(df):
    # For every dataset the regression target feature will be index = 1
    y = df[:, 1]
    X = np.delete(df, 1, axis=1)   # every column except column 1

    # We split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reduced = reduce_all(X_train, y_train, X_test, y_test)

    for name, (Xtr, ytr, Xte, yte) in reduced.items():
        print(name)
        #regression(Xtr, ytr, Xte, yte)

#Clustering application
def data_clust(df):
    # For every dataset the clustering target feature will be index = 1
    y = df[:, 2]
    X = np.delete(df, 2, axis=1)   # every column except column 1

    # We split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reduced = reduce_all(X_train, y_train, X_test, y_test)

    for name, (Xtr, ytr, Xte, yte) in reduced.items():
        print(name)
        #clustering(Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    iris = load_iris()  # Test dataset
    # Put the target in column 0, features after it, to match the convention
    data1 = np.column_stack([iris.target, iris.data])
    data_class(data1)