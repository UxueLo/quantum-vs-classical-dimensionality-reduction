import numpy as np
import umap
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
        "pca": (X_train_pca, y_train, X_test_pca, y_test),
        "umap": (X_train_umap, y_train, X_test_umap, y_test),
        "ae": (X_train_ae, y_train, X_test_ae, y_test),
        "z_map": (X_train_z_map, y_train, X_test_z_map, y_test),
    }

#Functions that receive a dataset and prepare the sets for training and testing before applying a traditional ML technique
#Fucntion that applies classification
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
        #classification(Xtr, ytr, Xte, yte)

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
    data1 = load_iris()  # Test dataset
    # Put the target in column 0, features after it, to match the convention
    data_class(data1)