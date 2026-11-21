import numpy as np
import umap
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import f_oneway, tukey_hsd
from scipy.stats import kruskal, shapiro, levene
from sklearn.model_selection import KFold
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
                                  RANSACRegressor, TheilSenRegressor, HuberRegressor)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             mean_squared_log_error, r2_score)

import warnings
from sklearn.datasets import load_diabetes
warnings.filterwarnings("ignore")

import os
import time
import matplotlib
matplotlib.use("Agg")   # save the figures as files without opening windows
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import plot_tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_curve

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

    # Sensitivity (TP/P) and specificity (TN/N). If a class doesn't appear in the test set
    # its value can't be computed (nan) and it is ignored in the average
    with np.errstate(divide="ignore", invalid="ignore"):
        if len(classes) == 2:
            # Binary problem: of the positive class (the second label)
            sensitivity = tp[1] / (tp[1] + fn[1])
            specificity = tn[1] / (tn[1] + fp[1])
        else:
            # Multiclass problem: of each class against the rest, then the macro average
            sensitivity = np.nanmean(tp / (tp + fn))
            specificity = np.nanmean(tn / (tn + fp))

    # ROC AUC and PR AUC need the probability of each class.
    # If the test set doesn't contain every class they can't be computed (nan)
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            pr_auc = average_precision_score(y_test, y_proba[:, 1], pos_label=classes[1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes)
            y_bin = label_binarize(y_test, classes=classes)
            pr_auc = average_precision_score(y_bin, y_proba, average="macro")
    except ValueError:
        roc_auc, pr_auc = np.nan, np.nan

    # Macro average: every class has the same weight (recommended in class when no class is more important).
    # The F1 is also given with the micro and weighted averages, to see how the averaging changes it
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "error_rate": 1 - accuracy_score(y_test, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "g_mean": np.sqrt(sensitivity * specificity),
        "precision": precision_score(y_test, y_pred, labels=classes, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, labels=classes, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, labels=classes, average="macro", zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, labels=classes, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, labels=classes, average="weighted", zero_division=0),
        "f0.5": fbeta_score(y_test, y_pred, beta=0.5, labels=classes, average="macro", zero_division=0),
        "f2": fbeta_score(y_test, y_pred, beta=2, labels=classes, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }


#Brier score: mean squared error between the predicted probabilities and the real class
#(0 = perfectly calibrated). Used to measure the probability calibration
def brier_score(y_test, y_proba, classes):
    y_onehot = (np.asarray(y_test).reshape(-1, 1) == np.asarray(classes).reshape(1, -1)).astype(float)
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


#Function that calibrates the probabilities of a model (03_b_Probability_Calibration_Example):
#Platt scaling (sigmoid) and isotonic regression. Returns the Brier score before and after
def calibration(clf, X_train, y_train, X_test, y_test):
    result = {"brier_uncalibrated": round(brier_score(y_test, clf.predict_proba(X_test), clf.classes_), 3)}
    _, counts = np.unique(y_train, return_counts=True)
    n_splits = int(min(5, counts.min()))
    for method in ["sigmoid", "isotonic"]:
        if n_splits < 2:
            result[f"brier_{method}"] = None
            continue
        calibrated = CalibratedClassifierCV(clone(clf), method=method, cv=n_splits)
        calibrated.fit(X_train, y_train)
        result[f"brier_{method}"] = round(brier_score(y_test, calibrated.predict_proba(X_test),
                                                      calibrated.classes_), 3)
    return result


#Function that draws the ROC and Precision-Recall curves of the best configuration of every model.
#In multiclass problems the micro average curve (all the classes together) is drawn
def plot_curves(title, results, y_test, folder="results"):
    os.makedirs(folder, exist_ok=True)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in results:
        classes, y_proba = r["classes"], r["y_proba"]
        if len(classes) == 2:
            y_true = (np.asarray(y_test) == classes[1]).astype(int)
            scores = y_proba[:, 1]
        else:
            y_true = label_binarize(y_test, classes=classes).ravel()
            scores = y_proba.ravel()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{r['model']} (AUC={r['test_metrics']['roc_auc']})")
        ax_pr.plot(recall, precision, label=f"{r['model']} (AUC={r['test_metrics']['pr_auc']})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax_roc.set(title=f"ROC curve - {title}", xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(title=f"Precision-Recall curve - {title}", xlabel="Recall", ylabel="Precision")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"curves_{title}.png"), dpi=120)
    plt.close(fig)


#Function that draws the best CART tree (03_b_CART notebook), useful to interpret the model
def plot_best_tree(title, clf, folder="results"):
    os.makedirs(folder, exist_ok=True)
    fig = plt.figure(figsize=(14, 7))
    plot_tree(clf, filled=True, max_depth=4, fontsize=7)
    plt.title(f"Best CART - {title}")
    fig.savefig(os.path.join(folder, f"tree_{title}.png"), dpi=120)
    plt.close(fig)


#Function that applies classification
def classification(X_train, y_train, X_test, y_test, title="dataset"):
    # Stratified cross-validation (recommended 10-fold). If a class has less than 10 examples,
    # the number of folds is reduced to that number so every fold has every class
    _, counts = np.unique(y_train, return_counts=True)
    n_splits = int(min(10, counts.min()))
    if n_splits < 2:
        raise ValueError("Every class needs at least 2 examples in the train set")
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Hyperparameter configurations we want to try for each model
    configs = []

    # CART (Decision Tree)
    #  - splitting criterion: Gini index (CART) or entropy / information gain (ID3)
    #  - prepruning: max_depth and min_samples_leaf stop the tree construction early
    #  - postpruning: ccp_alpha is the cost complexity pruning of CART
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

    # Support Vector Machines (SVM)
    # We evaluate the admissible kernels seen in theory: linear, rbf, poly, sigmoid
    for kernel in ["linear", "rbf", "poly", "sigmoid"]:
        for C_val in [0.1, 1.0, 10.0]:
            # The probability=True parameter is mandatory to compute roc_auc and pr_auc in evaluation().
            # max_iter limits the iterations: with some data (e.g. the poly kernel) the SVM can take
            # forever to converge. The data is standardized because the SVM depends on the scale
            clf = make_pipeline(StandardScaler(),
                                SVC(kernel=kernel, C=C_val, probability=True, random_state=42,
                                    max_iter=100000))
            hyperparameters = {"kernel": kernel, "C": C_val}
            configs.append(("SVM", clf, hyperparameters))

    # Random Forest
    # Ensemble method that creates multiple decision trees to reduce variance and overfitting
    #  - n_estimators: number of trees in the forest
    #  - max_features: feature randomness, number of features each tree can pick from ("sqrt" or all)
    #  - max_depth: prepruning of every tree
    for n_estimators in [50, 100, 200]:
        for max_depth in [None, 3, 5, 10]:
            for max_features in ["sqrt", None]:
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                             max_features=max_features, random_state=42)
                hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth,
                                   "max_features (feature randomness)": max_features}
                configs.append(("Random Forest", clf, hyperparameters))

    results = []
    for name, clf, hyperparameters in configs:
        # Accuracy of every fold, using only the train set
        accs = cross_val_score(clf, X_train, y_train, cv=kf)
        cv_acc = sum(accs) / len(accs)

        # Train with the whole train set and evaluate on test. Speed: training time
        start = time.perf_counter()
        clf = clf.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        metrics = evaluation(y_test, y_pred, y_proba, clf.classes_)

        results.append({"model": name, "hyperparameters": hyperparameters,
                        "cv_accuracy": round(float(cv_acc), 3),
                        "cv_accuracy_std": round(float(np.std(accs)), 3),
                        "train_time_s": round(train_time, 4),
                        "cv_scores": accs,
                        "test_metrics": {k: (round(float(v), 3) if k != "confusion_matrix" else v)
                                         for k, v in metrics.items()},
                        "clf": clf, "y_proba": y_proba, "classes": clf.classes_})

    # Best hyperparameter configuration of each model according to the cross validation
    best = {}
    for r in results:
        if r["model"] not in best or r["cv_accuracy"] > best[r["model"]]["cv_accuracy"]:
            best[r["model"]] = r

    # Probability calibration of the best configuration of each model
    for r in best.values():
        r["calibration"] = calibration(r["clf"], X_train, y_train, X_test, y_test)

    # Curves (ROC and Precision-Recall) and the best tree, saved as images in the "results" folder
    plot_curves(title, list(best.values()), y_test)
    plot_best_tree(title, best["CART"]["clf"])

    return list(best.values())


def data_class(df):
    # For every dataset the classification target feature will be index = 0
    y = df[:, 0]
    X = df[:, 1:].astype(float)

    # Stratified split: train and test keep the same class proportions (recommended in class).
    # It can't be used if a class has only 1 example
    _, counts = np.unique(y, return_counts=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if counts.min() >= 2 else None
    )

    reduced = reduce_all(X_train, y_train, X_test, y_test)

    metric_names = ["accuracy", "error_rate", "sensitivity", "specificity", "g_mean", "precision",
                    "recall", "f1", "f1_micro", "f1_weighted", "f0.5", "f2", "mcc", "roc_auc", "pr_auc"]
    best_cv = {}
    summary = []
    for name, (Xtr, ytr, Xte, yte) in reduced.items():
        print(f"\n########## {name} ##########")
        results = classification(Xtr, ytr, Xte, yte, title=name)

        # Table: best configuration of every model and the value of every metric on the test set
        rows = []
        for r in results:
            row = {"model": r["model"], "cv_accuracy": r["cv_accuracy"]}
            row.update({m: r["test_metrics"][m] for m in metric_names})
            rows.append(row)
        print_table(rows, ["model", "cv_accuracy"] + metric_names)

        # Statistical tests: are the differences between the models of this reduction significant?
        result = statistical_tests({r["model"]: r["cv_scores"] for r in results})
        print(f"\nStatistical tests between models: {result['conclusion']}"
              + (f" -> {', '.join(result['different_pairs'])}" if result.get("different_pairs") else ""))

        # Probability calibration (Brier score, lower is better) and training time
        print("\nCalibration (Brier score) and training time:")
        cal_rows = [{"model": r["model"], **r["calibration"], "train_time_s": r["train_time_s"]}
                    for r in results]
        print_table(cal_rows, ["model", "brier_uncalibrated", "brier_sigmoid", "brier_isotonic", "train_time_s"])

        # Hyperparameters and confusion matrix of every model
        print("\nBest hyperparameters and confusion matrix:")
        for r in results:
            print(f"   {r['model']}: {r['hyperparameters']}  confusion_matrix={r['test_metrics']['confusion_matrix']}")

        # Best model of this reduction method (highest cross validation accuracy)
        best_overall = max(results, key=lambda r: r["cv_accuracy"])
        best_cv[name] = best_overall["cv_scores"]
        print(f"\nBest model: {best_overall['model']} {best_overall['hyperparameters']} "
              f"(cv_accuracy={best_overall['cv_accuracy']}, test_f1={best_overall['test_metrics']['f1']})")
        summary.append({"reduction": name, "best_model": best_overall["model"],
                        "cv_accuracy": best_overall["cv_accuracy"],
                        "test_accuracy": best_overall["test_metrics"]["accuracy"],
                        "test_f1": best_overall["test_metrics"]["f1"],
                        "test_mcc": best_overall["test_metrics"]["mcc"]})

    # Final table: the best model of every reduction method, to compare them
    print("\n=== Comparison of the reduction methods (best model of each one) ===")
    print_table(summary, ["reduction", "best_model", "cv_accuracy", "test_accuracy", "test_f1", "test_mcc"])

    # Statistical test between reduction methods, using the best model of each one
    print("\n=== Statistical test between reduction methods (best model of each one, cross validation accuracy) ===")
    result = statistical_tests(best_cv)
    print(f"   {result['conclusion']}"
          + (f" -> {', '.join(result['different_pairs'])}" if result.get("different_pairs") else ""))


#Function that computes all the regression metrics seen in class (04_a_regression, "Evaluating Regression Models")
def evaluation_regression(y_test, y_pred, n_features):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if n > 1 else np.nan

    # MAPE: error as a percentage of the real value. It can't be used if a real value is 0
    mape = mean_absolute_percentage_error(y_test, y_pred) if np.all(np.abs(y_test) > 1e-12) else np.nan

    # MSLE: only defined when the real and predicted values are not negative
    msle = mean_squared_log_error(y_test, y_pred) if (y_test.min() >= 0 and y_pred.min() >= 0) else np.nan

    # Adjusted R2: penalizes the R2 when the model uses many features (n = samples, p = features)
    p = n_features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": mape,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "msle": msle,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
    }


#Function that checks if the differences between several groups of cross validation scores are
#statistically significant (03_c_Model_Evaluation): ANOVA + Tukey's HSD, or Kruskal-Wallis if the
#ANOVA assumptions (normality and homogeneity of variances) are not met.
#groups = {name: [score of every fold]}
def statistical_tests(groups, alpha=0.05):
    names = list(groups.keys())
    values = [np.asarray(groups[n], dtype=float) for n in names]

    if len(values) < 2:
        return {"test": None, "conclusion": "At least 2 groups are needed"}

    # If every fold of every group has the same score, there is nothing to test
    if np.allclose(np.concatenate(values), values[0][0]):
        return {"test": None, "conclusion": "All the groups have exactly the same scores"}

    # ANOVA assumptions: normality (Shapiro-Wilk) and homogeneity of variances (Levene)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normal = all(np.ptp(v) > 0 and shapiro(v).pvalue > alpha for v in values)
        equal_var = levene(*values).pvalue > alpha

    if normal and equal_var:
        test, p_value = "ANOVA", f_oneway(*values).pvalue
    else:
        # Non-parametric alternative seen in class
        test, p_value = "Kruskal-Wallis", kruskal(*values).pvalue

    result = {"test": test, "p_value": round(float(p_value), 4)}
    if p_value < alpha:
        result["conclusion"] = "At least one group is significantly different"
        # Post-hoc test seen in class: which specific pairs are different
        tukey = tukey_hsd(*values)
        result["different_pairs"] = [f"{names[i]} vs {names[j]} (p={tukey.pvalue[i, j]:.4f})"
                                     for i in range(len(names)) for j in range(i + 1, len(names))
                                     if tukey.pvalue[i, j] < alpha]
    else:
        result["conclusion"] = "No significant differences"
    return result


#Function that applies regression
def regression(X_train, y_train, X_test, y_test):
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    # 10-fold cross validation (in regression there are no classes, so it can't be stratified)
    n_splits = int(min(10, len(y_train)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    # Smallest train set inside a fold (KNN can't use more neighbors than that)
    min_fold_train = len(y_train) - int(np.ceil(len(y_train) / n_splits))

    # MAD of the target, used to give RANSAC thresholds that adapt to any dataset
    mad = np.median(np.abs(y_train - np.median(y_train)))

    # Some models need the features (and the target) to have a similar scale
    def scaled(model):
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), model),
                                          transformer=StandardScaler())

    # Hyperparameter configurations we want to try for each model
    configs = []

    # Ordinary Least Squares (OLS): no hyperparameters
    configs.append(("OLS", LinearRegression(), {}))

    # Ridge regression (L2 regularization). alpha = strength of the regularization
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        configs.append(("Ridge", Ridge(alpha=alpha), {"alpha (L2 regularization)": alpha}))

    # Lasso regression (L1 regularization, sets some coefficients to zero)
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        configs.append(("Lasso", Lasso(alpha=alpha, max_iter=10000),
                        {"alpha (L1 regularization)": alpha}))

    # Elastic Net: mix of Ridge and Lasso. l1_ratio = proportion of L1
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            configs.append(("Elastic Net", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
                            {"alpha": alpha, "l1_ratio": l1_ratio}))

    # Stochastic Gradient Descent (SGD) regression
    for penalty in ["l2", "l1", "elasticnet"]:
        for alpha in [0.0001, 0.001, 0.01]:
            configs.append(("SGD", scaled(SGDRegressor(penalty=penalty, alpha=alpha, max_iter=5000,
                                                       random_state=42)),
                            {"penalty": penalty, "alpha": alpha}))

    # Polynomial regression: Polynomial Features + a linear regression (04_b notebook).
    # degree = degree of the polynomial (usually not greater than 3 or 4, to avoid overfitting)
    for degree in [2, 3, 4]:
        for alpha in [0.0, 1.0]:
            linear = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
            model = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), linear)
            configs.append(("Polynomial Regression", model,
                            {"degree": degree, "linear_model": "OLS" if alpha == 0.0 else f"Ridge(alpha={alpha})"}))

    # Robust regression 1: RANSAC combined with OLS (04_c notebook)
    #  - max_trials: number of iterations
    #  - min_samples: minimum proportion of randomly chosen training examples
    #  - residual_threshold: max distance to the line to be an inlier (None = MAD, the default)
    for min_samples in [0.5, 0.95]:
        for threshold_factor in [None, 0.5, 2.0]:
            threshold = None if threshold_factor is None or mad == 0 else threshold_factor * mad
            configs.append(("RANSAC", RANSACRegressor(LinearRegression(), max_trials=100,
                                                      min_samples=min_samples,
                                                      residual_threshold=threshold, random_state=42),
                            {"max_trials": 100, "min_samples": min_samples,
                             "residual_threshold": "MAD (default)" if threshold is None
                             else f"{threshold_factor} x MAD"}))

    # Robust regression 2: Theil-Sen. max_subpopulation limits the combinations (and the runtime)
    for max_subpopulation in [1000, 10000]:
        configs.append(("Theil-Sen", TheilSenRegressor(max_subpopulation=max_subpopulation, random_state=42),
                        {"max_subpopulation": max_subpopulation}))

    # Robust regression 3: Huber. epsilon = threshold where the loss changes from MSE to MAE
    # (1.35 is the value recommended in class: 95% efficiency with normal errors)
    for epsilon in [1.35, 1.5, 1.75, 1.9]:
        for alpha in [0.0001, 0.01]:
            configs.append(("Huber", HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000),
                            {"epsilon": epsilon, "alpha (L2 regularization)": alpha}))

    # Additional regression methods in sklearn seen in class
    # Decision Tree regression (prepruning with max_depth and min_samples_leaf)
    for max_depth in [3, 5, 10, None]:
        for min_samples_leaf in [1, 5]:
            configs.append(("Decision Tree", DecisionTreeRegressor(max_depth=max_depth,
                                                                   min_samples_leaf=min_samples_leaf,
                                                                   random_state=42),
                            {"max_depth (prepruning)": max_depth,
                             "min_samples_leaf (prepruning)": min_samples_leaf}))

    # Random Forest regression
    for n_estimators in [50, 100]:
        for max_depth in [None, 5, 10]:
            configs.append(("Random Forest", RandomForestRegressor(n_estimators=n_estimators,
                                                                   max_depth=max_depth,
                                                                   random_state=42, n_jobs=-1),
                            {"n_estimators": n_estimators, "max_depth": max_depth}))

    # Support Vector Regression (SVR)
    for kernel in ["linear", "rbf", "poly"]:
        for C_val in [0.1, 1.0, 10.0]:
            configs.append(("SVR", scaled(SVR(kernel=kernel, C=C_val)), {"kernel": kernel, "C": C_val}))

    # Nearest neighbor regression (KNN)
    for n_neighbors in [3, 5, 10]:
        if n_neighbors > min_fold_train:
            continue
        for weights in ["uniform", "distance"]:
            configs.append(("KNN", scaled(KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)),
                            {"n_neighbors": n_neighbors, "weights": weights}))

    results = []
    for name, model, hyperparameters in configs:
        # RMSE of every fold, using only the train set (the main metric in the 04_g notebook).
        # If a configuration fails in some fold (e.g. RANSAC can't find inliers) it is discarded
        scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring="neg_root_mean_squared_error", error_score=np.nan)
        rmses = -scores
        if np.any(np.isnan(rmses)):
            continue

        # Train with the whole train set and evaluate on test
        try:
            model = model.fit(X_train, y_train)
        except ValueError:
            continue
        y_pred = model.predict(X_test)
        metrics = evaluation_regression(y_test, y_pred, n_features=X_train.shape[1])

        results.append({"model": name, "hyperparameters": hyperparameters,
                        "cv_rmse": round(float(np.mean(rmses)), 3),
                        "cv_rmse_std": round(float(np.std(rmses)), 3),
                        "cv_scores": rmses,
                        "test_metrics": {k: round(float(v), 3) for k, v in metrics.items()}})

    # Best hyperparameter configuration of each model according to the cross validation
    # (in regression the best one is the one with the LOWEST error)
    best = {}
    for r in results:
        if r["model"] not in best or r["cv_rmse"] < best[r["model"]]["cv_rmse"]:
            best[r["model"]] = r

    return list(best.values())

#Function that prints a list of results as a table (one row per model, one column per metric)
def print_table(rows, columns):
    widths = [max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in columns]
    print("  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(columns, widths)))


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

    metric_names = ["mae", "mape", "mse", "rmse", "msle", "r2", "adjusted_r2"]
    best_cv = {}
    summary = []
    for name, (Xtr, ytr, Xte, yte) in reduced.items():
        print(f"\n########## {name} ##########")
        results = regression(Xtr, ytr, Xte, yte)

        # Table: best configuration of every model and the value of every metric on the test set
        rows = []
        for r in results:
            row = {"model": r["model"], "cv_rmse": r["cv_rmse"]}
            row.update({m: r["test_metrics"][m] for m in metric_names})
            rows.append(row)
        print_table(rows, ["model", "cv_rmse"] + metric_names)

        # Statistical tests: are the differences between the models of this reduction significant?
        result = statistical_tests({r["model"]: r["cv_scores"] for r in results})
        print(f"\nStatistical tests between models: {result['conclusion']}"
              + (f" -> {', '.join(result['different_pairs'])}" if result.get("different_pairs") else ""))

        # Hyperparameters chosen for every model
        print("\nBest hyperparameters:")
        for r in results:
            print(f"   {r['model']}: {r['hyperparameters']}")

        # Best model of this reduction method (lowest cross validation RMSE)
        best_overall = min(results, key=lambda r: r["cv_rmse"])
        best_cv[name] = best_overall["cv_scores"]
        print(f"\nBest model: {best_overall['model']} {best_overall['hyperparameters']} "
              f"(cv_rmse={best_overall['cv_rmse']}, test_r2={best_overall['test_metrics']['r2']})")
        summary.append({"reduction": name, "best_model": best_overall["model"],
                        "cv_rmse": best_overall["cv_rmse"],
                        "test_rmse": best_overall["test_metrics"]["rmse"],
                        "test_r2": best_overall["test_metrics"]["r2"]})

    # Final table: the best model of every reduction method, to compare them
    print("\n=== Comparison of the reduction methods (best model of each one) ===")
    print_table(summary, ["reduction", "best_model", "cv_rmse", "test_rmse", "test_r2"])

    # Statistical test between reduction methods, using the best model of each one
    print("\n=== Statistical test between reduction methods (best model of each one, cross validation RMSE) ===")
    result = statistical_tests(best_cv)
    print(f"   {result['conclusion']}"
          + (f" -> {', '.join(result['different_pairs'])}" if result.get("different_pairs") else ""))

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
    data1 = np.column_stack([iris.target, iris.data])
    data_class(data1)

    diabetes = load_diabetes()

    data2 = np.column_stack([diabetes.data[:, 0], diabetes.target, diabetes.data[:, 1:]])
    data_regr(data2)