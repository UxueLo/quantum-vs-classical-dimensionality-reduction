import numpy as np
import umap
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA, PCA
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
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             mean_squared_log_error, r2_score)

import warnings
from sklearn.datasets import load_diabetes
warnings.filterwarnings("ignore")

import os
import csv
import time
import matplotlib
matplotlib.use("Agg")   # save the figures as files without opening windows
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import plot_tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_curve
import optuna # Automatic hyperparameter optimization
optuna.logging.set_verbosity(optuna.logging.WARNING) # one line per trial would flood the console

# Optuna configuration
N_TRIALS = 30 # number of trials (hyperparameter configurations tried) per model
OPTUNA_SEED = 42 # seed of the sampler: makes the search reproducible between runs
SAVE_OPTUNA_PLOTS = True # To save the optimization history and parameter importance of every study as HTML


def reduce_all(X_train, y_train, X_test, y_test):
    
    # Number of dimensions we want after the reduction
    n_qubits = 4

    # PCA (linear). It's fit only on train
    pca = PCA(n_components=n_qubits)
    X_train_linear_pca = pca.fit_transform(X_train)
    X_test_linear_pca = pca.transform(X_test)

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
        "PCA": (X_train_linear_pca, y_train, X_test_linear_pca, y_test),
        "KernelPCA": (X_train_pca, y_train, X_test_pca, y_test),
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

#Scores used to compute ROC AUC, PR AUC and the curves. The predict_proba of the SVM is computed with an
#internal cross validation (Platt scaling) and sklearn warns it can be inconsistent with predict(): with some
#hyperparameters the probabilities are inverted (ROC AUC close to 0 with high accuracy). For models with a
#decision_function (the SVM) its scores are used instead, because they always agree with the predictions.
#They are converted to values between 0 and 1 that sum 1 per row (softmax), as roc_auc_score needs in multiclass
def ranking_scores(clf, X, y_proba):
    if not hasattr(clf, "decision_function"):
        return y_proba
    d = np.asarray(clf.decision_function(X), dtype=float)
    if d.ndim == 1:
        # Binary problem: one score per example, positive means the second class
        return np.column_stack([-d, d])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

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


#Function that saves a list of results as a CSV file (one row per configuration), in the "results" folder
def save_csv(rows, columns, filename, folder="results"):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

#Conditional search space for the maxi depth of a tree
#Optuna decides if the depth is limited at all (None = unlimited) and, only in that case, which depth to use
def suggest_max_depth(trial, low=2, high=20):
    if trial.suggest_categorical("limit_depth", [True, False]):
        return trial.suggest_int("max_depth", low, high)
    return None


#Function that saves the Optuna visualizations of a study as HTML files 
def save_optuna_plots(study, name, folder="results"):
    try:
        import optuna.visualization as vis
    except ImportError:
        return
    os.makedirs(folder, exist_ok=True)
    safe_name = name.replace(" ", "_")
    for plot_name, plot_function in [("history", vis.plot_optimization_history),
                                     ("importance", vis.plot_param_importances)]:
        try:
            plot_function(study).write_html(os.path.join(folder, f"optuna_{plot_name}_{safe_name}.html"))
        except Exception:
            # plotly not installed, or not enough trials / parameters to compute the plot
            pass


#Function that optimizes the hyperparameters of ONE model with Optuna and returns the result of every completed trial
def run_optuna_search(model_name, space, evaluate_trial, direction, n_trials, study_name):
    results = []

    def objective(trial):
        model, hyperparameters = space(trial)
        score, result = evaluate_trial(model, hyperparameters)
        result["model"] = model_name
        results.append(result)
        return score

    study = optuna.create_study(direction=direction, study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)

    if SAVE_OPTUNA_PLOTS and results:
        save_optuna_plots(study, study_name)
    return results


#Function that applies classification
def classification(X_train, y_train, X_test, y_test, title="dataset"):
    # Stratified cross-validation (recommended 10-fold). If a class has less than 10 examples,
    # the number of folds is reduced to that number so every fold has every class
    _, counts = np.unique(y_train, return_counts=True)
    n_splits = int(min(10, counts.min()))
    if n_splits < 2:
        raise ValueError("Every class needs at least 2 examples in the train set")
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Hyperparameter search space of each model. Instead of a fixed grid, Optuna (TPE sampler)
    # decides which configuration to try next, using the results of the previous trials

    # CART (Decision Tree)
    #  - splitting criterion: Gini index (CART) or entropy / information gain (ID3)
    #  - prepruning: max_depth and min_samples_leaf stop the tree construction early
    #  - postpruning: ccp_alpha is the cost complexity pruning of CART
    def space_cart(trial):
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = suggest_max_depth(trial, 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.1)
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     ccp_alpha=ccp_alpha, random_state=42)
        hyperparameters = {"splitting_criterion": criterion,
                           "max_depth (prepruning)": max_depth,
                           "min_samples_leaf (prepruning)": min_samples_leaf,
                           "cost_complexity_alpha (postpruning)": round(ccp_alpha, 5)}
        return clf, hyperparameters

    # Gaussian Naive Bayes
    #  - var_smoothing: the small positive variance floor used when a feature
    #    has zero variance inside a class
    def space_naive_bayes(trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)
        clf = GaussianNB(var_smoothing=var_smoothing)
        hyperparameters = {"variance_floor (var_smoothing)": var_smoothing}
        return clf, hyperparameters

    # Support Vector Machines (SVM)
    # We evaluate the admissible kernels seen in theory: linear, rbf, poly, sigmoid
    #  - C: regularization strength (log scale)
    #  - gamma: kernel coefficient (log scale). Not used by the linear kernel, so it is conditional
    #  - degree: degree of the polynomial, only for the poly kernel
    def space_svm(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        C_val = trial.suggest_float("C", 1e-2, 1e2, log=True)
        hyperparameters = {"kernel": kernel, "C": round(C_val, 5)}
        svc_params = {}
        if kernel != "linear":
            svc_params["gamma"] = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
            hyperparameters["gamma"] = round(svc_params["gamma"], 5)
        if kernel == "poly":
            svc_params["degree"] = trial.suggest_int("degree", 2, 5)
            hyperparameters["degree"] = svc_params["degree"]
        # The probability=True parameter is mandatory to compute roc_auc and pr_auc in evaluation().
        # max_iter limits the iterations: with some data (e.g. the poly kernel) the SVM can take
        # forever to converge. The data is standardized because the SVM depends on the scale
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel=kernel, C=C_val, probability=True, random_state=42,
                                max_iter=100000, **svc_params))
        return clf, hyperparameters

    # Random Forest
    # Ensemble method that creates multiple decision trees to reduce variance and overfitting
    #  - n_estimators: number of trees in the forest
    #  - max_features: feature randomness, number of features each tree can pick from ("sqrt", "log2" or all)
    #  - max_depth: prepruning of every tree
    def space_random_forest(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = suggest_max_depth(trial, 2, 15)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     max_features=max_features, random_state=42)
        hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth,
                           "max_features (feature randomness)": max_features}
        return clf, hyperparameters

    spaces = [("CART", space_cart),
              ("Gaussian Naive Bayes", space_naive_bayes),
              ("SVM", space_svm),
              ("Random Forest", space_random_forest)]

    # Optuna optimizes the mean accuracy of the cross validation, using only the train set
    def evaluate_trial(clf, hyperparameters):
        # Accuracy of every fold, using only the train set
        accs = cross_val_score(clf, X_train, y_train, cv=kf)
        cv_acc = sum(accs) / len(accs)

        # Train with the whole train set and evaluate on test
        start = time.perf_counter()
        clf = clf.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        y_pred = clf.predict(X_test)
        y_proba = ranking_scores(clf, X_test, clf.predict_proba(X_test))
        metrics = evaluation(y_test, y_pred, y_proba, clf.classes_)

        result = {"hyperparameters": hyperparameters,
                  "cv_accuracy": round(float(cv_acc), 3),
                  "cv_accuracy_std": round(float(np.std(accs)), 3),
                  "train_time_s": round(train_time, 4),
                  "cv_scores": accs,
                  "test_metrics": {k: (round(float(v), 3) if k != "confusion_matrix" else v)
                                   for k, v in metrics.items()},
                  "clf": clf, "y_proba": y_proba, "classes": clf.classes_}
        return float(cv_acc), result

    # One Optuna study per model to maximize the cross validation accuracy
    results = []
    for name, space in spaces:
        results += run_optuna_search(name, space, evaluate_trial, direction="maximize",
                                     n_trials=N_TRIALS, study_name=f"classification_{title}_{name}")

    # Table with every configuration tried (one row per Optuna trial), saved as a CSV in the "results" folder
    save_csv([{"model": r["model"], "hyperparameters": r["hyperparameters"], "cv_accuracy": r["cv_accuracy"],
               "cv_accuracy_std": r["cv_accuracy_std"], "train_time_s": r["train_time_s"], **r["test_metrics"]}
              for r in results],
             ["model", "hyperparameters", "cv_accuracy", "cv_accuracy_std", "train_time_s", "accuracy",
              "error_rate", "sensitivity", "specificity", "g_mean", "precision", "recall", "f1", "f1_micro",
              "f1_weighted", "f0.5", "f2", "mcc", "roc_auc", "pr_auc"],
             f"classification_all_configs_{title}.csv")

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

        # Hyperparameters (found by Optuna) and confusion matrix of every model
        print("\nBest hyperparameters (Optuna) and confusion matrix:")
        for r in results:
            print(f"   {r['model']}: {r['hyperparameters']}  confusion_matrix={r['test_metrics']['confusion_matrix']}")

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
def regression(X_train, y_train, X_test, y_test, title="dataset"):
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

    # Instead of a fixed grid Optuna decides which configuration to try next using the results of the previous trials

    # Ordinary Least Squares (OLS): no hyperparameters (a single trial is enough)
    def space_ols(trial):
        return LinearRegression(), {}

    # Ridge regression (L2 regularization). alpha = strength of the regularization
    def space_ridge(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
        return Ridge(alpha=alpha), {"alpha (L2 regularization)": round(alpha, 5)}

    # Lasso regression (L1 regularization, sets some coefficients to zero)
    def space_lasso(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e1, log=True)
        return Lasso(alpha=alpha, max_iter=10000), {"alpha (L1 regularization)": round(alpha, 5)}

    # Elastic Net: mix of Ridge and Lasso. l1_ratio = proportion of L1
    def space_elastic_net(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        return (ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
                {"alpha": round(alpha, 5), "l1_ratio": round(l1_ratio, 3)})

    # Stochastic Gradient Descent (SGD) regression. l1_ratio only exists with the elasticnet penalty
    def space_sgd(trial):
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        sgd_params = {}
        hyperparameters = {"penalty": penalty, "alpha": round(alpha, 6)}
        if penalty == "elasticnet":
            sgd_params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
            hyperparameters["l1_ratio"] = round(sgd_params["l1_ratio"], 3)
        model = scaled(SGDRegressor(penalty=penalty, alpha=alpha, max_iter=5000, random_state=42,
                                    **sgd_params))
        return model, hyperparameters

    # Polynomial regression: Polynomial Features + a linear regression (04_b notebook).
    # degree = degree of the polynomial (usually not greater than 3 or 4, to avoid overfitting)
    def space_polynomial(trial):
        degree = trial.suggest_int("degree", 2, 4)
        linear_model = trial.suggest_categorical("linear_model", ["OLS", "Ridge"])
        if linear_model == "Ridge":
            alpha = trial.suggest_float("ridge_alpha", 1e-2, 1e2, log=True)
            linear, label = Ridge(alpha=alpha), f"Ridge(alpha={alpha:.4g})"
        else:
            linear, label = LinearRegression(), "OLS"
        model = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), linear)
        return model, {"degree": degree, "linear_model": label}

    # Robust regression 1: RANSAC combined with OLS (04_c notebook)
    #  - max_trials: number of iterations
    #  - min_samples: minimum proportion of randomly chosen training examples
    #  - residual_threshold: max distance to the line to be an inlier (None = MAD, the default)
    def space_ransac(trial):
        max_trials = trial.suggest_int("max_trials", 50, 300, step=50)
        min_samples = trial.suggest_float("min_samples", 0.3, 0.99)
        if trial.suggest_categorical("default_threshold", [True, False]) or mad == 0:
            threshold, threshold_label = None, "MAD (default)"
        else:
            threshold_factor = trial.suggest_float("threshold_factor", 0.1, 5.0, log=True)
            threshold, threshold_label = threshold_factor * mad, f"{threshold_factor:.3g} x MAD"
        model = RANSACRegressor(LinearRegression(), max_trials=max_trials, min_samples=min_samples,
                                residual_threshold=threshold, random_state=42)
        return model, {"max_trials": max_trials, "min_samples": round(min_samples, 3),
                       "residual_threshold": threshold_label}

    # Robust regression 2: Theil-Sen. max_subpopulation limits the combinations (and the runtime)
    def space_theil_sen(trial):
        max_subpopulation = trial.suggest_int("max_subpopulation", 500, 10000, log=True)
        return (TheilSenRegressor(max_subpopulation=max_subpopulation, random_state=42),
                {"max_subpopulation": max_subpopulation})

    # Robust regression 3: Huber. epsilon = threshold where the loss changes from MSE to MAE
    # (1.35 is the value recommended in class: 95% efficiency with normal errors). It must be > 1
    def space_huber(trial):
        epsilon = trial.suggest_float("epsilon", 1.1, 3.0)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        return (HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000),
                {"epsilon": round(epsilon, 3), "alpha (L2 regularization)": round(alpha, 6)})

    # Additional regression methods in sklearn seen in class
    # Decision Tree regression (prepruning with max_depth and min_samples_leaf)
    def space_decision_tree(trial):
        max_depth = suggest_max_depth(trial, 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                      random_state=42)
        return model, {"max_depth (prepruning)": max_depth,
                       "min_samples_leaf (prepruning)": min_samples_leaf}

    # Random Forest regression
    def space_random_forest(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
        max_depth = suggest_max_depth(trial, 2, 15)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=42, n_jobs=-1)
        return model, {"n_estimators": n_estimators, "max_depth": max_depth}

    # Support Vector Regression (SVR). gamma is not used by the linear kernel, degree only by poly
    def space_svr(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        C_val = trial.suggest_float("C", 1e-2, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        hyperparameters = {"kernel": kernel, "C": round(C_val, 5), "epsilon": round(epsilon, 5)}
        svr_params = {}
        if kernel != "linear":
            svr_params["gamma"] = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
            hyperparameters["gamma"] = round(svr_params["gamma"], 5)
        if kernel == "poly":
            svr_params["degree"] = trial.suggest_int("degree", 2, 4)
            hyperparameters["degree"] = svr_params["degree"]
        return scaled(SVR(kernel=kernel, C=C_val, epsilon=epsilon, **svr_params)), hyperparameters

    # Nearest neighbor regression (KNN). It can't use more neighbors than the smallest train fold
    def space_knn(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, max(1, min(30, min_fold_train)))
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        model = scaled(KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))
        return model, {"n_neighbors": n_neighbors, "weights": weights}
    
    # Quantile regression: predicts a quantile of the target instead of the mean (more robust to outliers)
    #  - quantile: which quantile is predicted (0.5 = the median)
    #  - alpha: L1 regularization
    def space_quantile(trial):
        quantile = trial.suggest_float("quantile", 0.25, 0.75)
        alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
        model = scaled(QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs"))
        return model, {"quantile": round(quantile, 3), "alpha (L1 regularization)": round(alpha, 6)}

    # Gaussian Process regression
    #  - kernel: covariance function (similarity between examples). Its length scale is fitted automatically
    #  - alpha: noise added to the diagonal (regularization)
    def space_gaussian_process(trial):
        kernel_name = trial.suggest_categorical("kernel", ["RBF", "Matern 1.5", "Matern 2.5", "RationalQuadratic"])
        kernels = {"RBF": RBF(), "Matern 1.5": Matern(nu=1.5), "Matern 2.5": Matern(nu=2.5),
                   "RationalQuadratic": RationalQuadratic()}
        alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        model = scaled(GaussianProcessRegressor(kernel=kernels[kernel_name], alpha=alpha, random_state=42))
        return model, {"kernel": kernel_name, "alpha (noise)": round(alpha, 5)}
    
    # (model name, search space, number of trials)
    spaces = [("OLS", space_ols, 1),
              ("Ridge", space_ridge, N_TRIALS),
              ("Lasso", space_lasso, N_TRIALS),
              ("Elastic Net", space_elastic_net, N_TRIALS),
              ("SGD", space_sgd, N_TRIALS),
              ("Polynomial Regression", space_polynomial, N_TRIALS),
              ("RANSAC", space_ransac, N_TRIALS),
              ("Theil-Sen", space_theil_sen, N_TRIALS),
              ("Huber", space_huber, N_TRIALS),
              ("Decision Tree", space_decision_tree, N_TRIALS),
              ("Random Forest", space_random_forest, N_TRIALS),
              ("SVR", space_svr, N_TRIALS),
              ("KNN", space_knn, N_TRIALS),
              ("Quantile Regression", space_quantile, N_TRIALS),
              ("Gaussian Process", space_gaussian_process, N_TRIALS)]

    # Optuna optimizes the mean RMSE of the cross validation using only the train set
    def evaluate_trial(model, hyperparameters):
        # If a configuration fails in some fold (ex. RANSAC can't find inliers) the trial is pruned
        scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring="neg_root_mean_squared_error", error_score=np.nan)
        rmses = -scores
        if np.any(np.isnan(rmses)):
            raise optuna.TrialPruned()

        # Train with the whole train set and evaluate on test
        try:
            model = model.fit(X_train, y_train)
        except ValueError:
            raise optuna.TrialPruned()
        y_pred = model.predict(X_test)
        metrics = evaluation_regression(y_test, y_pred, n_features=X_train.shape[1])

        result = {"hyperparameters": hyperparameters,
                  "cv_rmse": round(float(np.mean(rmses)), 3),
                  "cv_rmse_std": round(float(np.std(rmses)), 3),
                  "cv_scores": rmses,
                  "test_metrics": {k: round(float(v), 3) for k, v in metrics.items()}}
        return float(np.mean(rmses)), result

    # One Optuna study per model to minimize the cross validation RMSE
    results = []
    for name, space, n_trials in spaces:
        results += run_optuna_search(name, space, evaluate_trial, direction="minimize",
                                     n_trials=n_trials, study_name=f"regression_{title}_{name}")

    # Table with every configuration tried (04_g notebook, one row per Optuna trial),
    # saved as a CSV in the "results" folder
    save_csv([{"model": r["model"], "hyperparameters": r["hyperparameters"], "cv_rmse": r["cv_rmse"],
               "cv_rmse_std": r["cv_rmse_std"], **r["test_metrics"]} for r in results],
             ["model", "hyperparameters", "cv_rmse", "cv_rmse_std", "mae", "mape", "mse", "rmse", "msle",
              "r2", "adjusted_r2"],
             f"regression_all_configs_{title}.csv")

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

    for name, (Xtr, ytr, Xte, yte) in reduced.items():
        print(f"\n########## {name} ##########")
        results = regression(Xtr, ytr, Xte, yte, title=name)

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

        # Hyperparameters chosen by Optuna for every model
        print("\nBest hyperparameters (Optuna):")
        for r in results:
            print(f"   {r['model']}: {r['hyperparameters']}")

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