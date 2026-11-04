from __future__ import annotations
 
import difflib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
 
logger = logging.getLogger("createCSV")
 
MODELO_GEMINI = "gemini-2.5-flash"
LAMBDA_REG = 1e-3
MAX_QUBITS_SIM = 10          # el simulador estadovector crece como 2^n
MAX_MUESTRAS = 200           # N máximo por dataset (los cálculos son O(N^3))
SEMILLA = 42
 
# ==============================================================================
# 0. ESQUEMA FINAL DEL CSV (todas las filas tienen TODAS estas columnas)
# ==============================================================================
 
COLUMNAS_FINALES: List[str] = [
    # --- identificación / jerarquía paper -> dataset -> experimento
    "experiment_id", "archivo_origen", "paper_title",
    "dataset_name", "dataset_local_key", "domain_type", "task_type",
    "experiment_label", "quantum_model", "classical_baseline",
    # --- configuración del experimento (extraída del paper)
    "encoding_type", "n_qubits", "circuit_depth", "is_real_hardware", "metric_name",
    # --- resultados empíricos (extraídos del paper)
    "score_classical", "score_quantum", "score_gap",
    "quantum_advantage_reported", "kernel_concentration_observed",
    # --- métricas nivel DATASET (calculadas, compartidas por sus experimentos)
    "n_samples", "n_features_original", "n_classes", "delta_linearity", "s_C_min",
    # --- métricas nivel EXPERIMENTO (calculadas con su K_Q)
    "k_latente", "eta_pca_k", "g_CQ_star", "s_Q", "psi_huang", "A_c_Q", "d_eff_Q",
    "n_qubits_simulados", "kernel_source",
    # --- trazabilidad de valores imputados/derivados
    "circuit_depth_imputed", "advantage_derived",
]
 
 
# ==============================================================================
# 1. ESQUEMAS PYDANTIC PARA GEMINI  (Paper -> Datasets -> Experimentos)
# ==============================================================================
 
class ExperimentoPaper(BaseModel):
    """Un experimento = un modelo/configuración evaluado sobre UN dataset concreto."""
    experiment_label: str = Field(
        description="Etiqueta corta que distinga este experimento de los otros del mismo dataset "
                    "(ej: 'QSVM ZZ reps=2 8q', 'PQK 12q', 'IBM hardware run').")
    quantum_model: str = Field(
        description="Modelo cuántico evaluado (QSVM, PQK, VQC, QNN, quantum kernel ridge...). "
                    "'unspecified' si no se indica.")
    classical_baseline: str = Field(
        description="Mejor modelo clásico de comparación (SVM-RBF, RandomForest, MLP...). "
                    "'unspecified' si no se indica.")
    encoding_type: str = Field(
        description="Mapa de características/encoding (ZZFeatureMap, Angle, IQP, Covariant, PQK...). "
                    "'unspecified' si no se indica.")
    n_qubits: Optional[int] = Field(default=None, description="Número de qubits usados en ESTE experimento.")
    circuit_depth: Optional[int] = Field(
        default=None, description="Profundidad o número de repeticiones (reps) del circuito en ESTE experimento.")
    is_real_hardware: Optional[bool] = Field(
        default=None, description="True si usaron QPU real; False si simulador o no se menciona.")
    metric_name: str = Field(
        description="Métrica de los scores (accuracy, F1, AUC, R2, RMSE...). 'unspecified' si no se indica.")
    score_classical: Optional[float] = Field(
        default=None, description="Score del modelo clásico en ESTE experimento. Si es un porcentaje, "
                                  "devuélvelo tal cual aparece (se normalizará después).")
    score_quantum: Optional[float] = Field(
        default=None, description="Score del modelo cuántico en ESTE experimento.")
    quantum_advantage_reported: Optional[bool] = Field(
        default=None, description="True si los autores concluyen ventaja cuántica en este experimento; "
                                  "None si no concluyen nada explícito.")
    kernel_concentration_observed: Optional[bool] = Field(
        default=None, description="True si mencionan concentración del kernel / barren plateaus en este "
                                  "experimento; False si no lo mencionan.")
 
 
class DatasetPaper(BaseModel):
    dataset_name: str = Field(description="Nombre del dataset tal como lo llama el paper.")
    local_dataset_key: Optional[str] = Field(
        default=None, description="Si el dataset coincide con uno de la lista de 'datasets locales "
                                  "disponibles' del prompt, copia aquí esa clave EXACTA; si no, null.")
    domain_type: str = Field(description="Dominio (medical_image, tabular_health, physics, credit, synthetic...).")
    task_type: str = Field(description="'classification', 'regression' o 'clustering'.")
    experimentos: List[ExperimentoPaper] = Field(
        description="TODOS los experimentos/modelos/configuraciones reportados sobre ESTE dataset.")
 
 
class ReportePaper(BaseModel):
    paper_title: Optional[str] = Field(default=None, description="Título del artículo.")
    datasets: List[DatasetPaper] = Field(description="Todos los datasets usados en el paper.")
 
 
# ==============================================================================
# 2. EXTRACCIÓN CON GEMINI (jerárquica) Y APLANADO A FILAS
# ==============================================================================
 
def _construir_prompt(datasets_disponibles: Optional[List[str]]) -> str:
    lista = ", ".join(datasets_disponibles) if datasets_disponibles else "(ninguno)"
    return f"""
Analiza este artículo de investigación sobre aprendizaje automático cuántico y extrae su
estructura completa: PAPER -> DATASETS -> EXPERIMENTOS.
 
Reglas:
1. Crea una entrada en 'datasets' por CADA dataset distinto usado (reales o sintéticos).
2. Dentro de cada dataset, crea un experimento independiente por CADA combinación
   modelo / encoding / nº de qubits / profundidad / simulador-vs-hardware con resultados propios.
   Un mismo dataset suele producir varios experimentos con scores distintos.
3. Los scores (score_classical, score_quantum) deben corresponder a ESE experimento concreto
   (tablas, figuras o texto). No mezcles resultados de datasets o configuraciones distintas.
4. NO inventes valores: si un número no aparece en el paper, déjalo en null.
5. Datasets locales disponibles para calcular métricas: [{lista}].
   Si un dataset del paper corresponde a uno de ellos, rellena 'local_dataset_key' con la clave exacta.
"""
 
 
def extraer_reporte_de_pdf(
    pdf_path: str,
    client_api: Optional[genai.Client] = None,
    datasets_disponibles: Optional[List[str]] = None,
    modelo: str = MODELO_GEMINI,
    reintentos: int = 3,
) -> Optional[ReportePaper]:
    """Llama a Gemini con el PDF y devuelve el reporte jerárquico validado."""
    path = Path(pdf_path)
    if client_api is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GEMINI_API_KEY (variable de entorno o archivo .env).")
        client_api = genai.Client(api_key=api_key)
 
    pdf_bytes = path.read_bytes()
    prompt = _construir_prompt(datasets_disponibles)
 
    for intento in range(1, reintentos + 1):
        try:
            response = client_api.models.generate_content(
                model=modelo,
                contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReportePaper,
                    temperature=0.1,
                ),
            )
            if not response.text:
                raise ValueError("Respuesta vacía de Gemini.")
            return ReportePaper.model_validate_json(response.text)
        except (ValidationError, ValueError, Exception) as e:  # red, cuota, JSON truncado...
            logger.warning("Intento %d/%d fallido para '%s': %s", intento, reintentos, path.name, e)
            if intento < reintentos:
                time.sleep(2 ** intento)
    logger.error("No se pudo extraer información de '%s'.", path.name)
    return None
 
 
def _es_metrica_de_error(metric_name: Optional[str]) -> bool:
    return bool(metric_name) and any(t in metric_name.lower() for t in ("error", "mse", "rmse", "mae", "loss"))
 
 
def _normalizar_score(valor: Optional[float], metric_name: Optional[str]) -> Optional[float]:
    """Convierte porcentajes (0-100) a fracción (0-1) para métricas tipo accuracy/F1/AUC."""
    if valor is None or not math.isfinite(valor):
        return None
    if not _es_metrica_de_error(metric_name) and 1.0 < valor <= 100.0:
        return valor / 100.0
    return float(valor)
 
 
def aplanar_reporte(reporte: ReportePaper, archivo: str) -> List[Dict[str, Any]]:
    """Paper -> datasets -> experimentos  ==>  lista plana (1 dict por experimento)."""
    registros: List[Dict[str, Any]] = []
    for i, ds in enumerate(reporte.datasets):
        for j, exp in enumerate(ds.experimentos):
            reg = {
                "archivo_origen": archivo,
                "paper_title": reporte.paper_title,
                "dataset_idx": i,
                "experiment_idx": j,
                "dataset_name": ds.dataset_name,
                "local_dataset_key_sugerida": ds.local_dataset_key,
                "domain_type": ds.domain_type,
                "task_type_paper": ds.task_type,
            }
            reg.update(exp.model_dump())
            registros.append(reg)
    return registros
 
 
# ==============================================================================
# 3. REGISTRO DE DATASETS LOCALES (puente entre el paper y los cálculos)
# ==============================================================================
 
def _norm_nombre(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())
 
 
@dataclass
class DatasetLocal:
    key: str
    X: np.ndarray                      # (N, d) crudo, ya submuestreado
    y: np.ndarray                      # (N,) etiquetas codificadas (enteros) o floats
    task_type: str                     # 'classification' | 'regression'
    # cachés (se rellenan bajo demanda)
    X_std: Optional[np.ndarray] = None
    K_C_list: Optional[List[np.ndarray]] = None
    base: Optional[Dict[str, float]] = field(default=None)
 
    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y))) if self.task_type == "classification" else 0
 
 
class RegistroDatasets:
    """
    Asocia el nombre de un dataset del paper con sus datos locales (X, y).
 
        reg = RegistroDatasets()
        reg.registrar("breast_cancer", (X, y), aliases=["Wisconsin Breast Cancer"])
        reg.registrar("mi_dataset", lambda: cargar(), aliases=[...])      # carga perezosa
        reg.registrar_csv("german_credit", "data/german.csv", target_col="class")
    """
 
    def __init__(self, max_muestras: int = MAX_MUESTRAS, semilla: int = SEMILLA):
        self.max_muestras = max_muestras
        self.semilla = semilla
        self._fuentes: Dict[str, Union[Tuple[np.ndarray, np.ndarray], Callable]] = {}
        self._alias: Dict[str, str] = {}          # nombre normalizado -> clave
        self._cache: Dict[str, DatasetLocal] = {}
 
    def claves(self) -> List[str]:
        return list(self._fuentes)
 
    def registrar(self, key: str, datos, aliases: Iterable[str] = ()):
        self._fuentes[key] = datos
        for nombre in (key, *aliases):
            self._alias[_norm_nombre(nombre)] = key
        self._cache.pop(key, None)
 
    def registrar_csv(self, key: str, ruta: str, target_col: str, aliases: Iterable[str] = ()):
        def _loader():
            df = pd.read_csv(ruta).dropna()
            return df.drop(columns=[target_col]).select_dtypes("number").to_numpy(float), df[target_col].to_numpy()
        self.registrar(key, _loader, aliases)
 
    def resolver(self, *candidatos: Optional[str]) -> Optional[str]:
        """Devuelve la clave local que corresponde a alguno de los nombres dados (o None)."""
        normalizados = [_norm_nombre(c) for c in candidatos if c]
        for n in normalizados:                                   # 1) exacto / alias
            if n in self._alias:
                return self._alias[n]
        for n in normalizados:                                   # 2) contención ("breastcancer" en "breastcancerwisconsin")
            for alias_n, key in self._alias.items():
                if len(alias_n) >= 4 and (alias_n in n or n in alias_n) and len(n) >= 4:
                    return key
        for n in normalizados:                                   # 3) aproximado
            m = difflib.get_close_matches(n, list(self._alias), n=1, cutoff=0.85)
            if m:
                return self._alias[m[0]]
        return None
 
    def obtener(self, key: str) -> DatasetLocal:
        if key in self._cache:
            return self._cache[key]
        fuente = self._fuentes[key]
        X, y = fuente() if callable(fuente) else fuente
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"Dataset '{key}': X debe ser (N,d) e y de longitud N.")
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
 
        es_clasif = (not np.issubdtype(y.dtype, np.floating)) or (
            np.all(np.mod(y, 1) == 0) and len(np.unique(y)) <= 10)
        task = "classification" if es_clasif else "regression"
        y = LabelEncoder().fit_transform(y) if task == "classification" else y.astype(float)
        y = np.asarray(y) # Esto le confirma a Pylance que es un ndarray
 
        if len(y) > self.max_muestras:                           # submuestreo reproducible
            estrat = y if task == "classification" and np.bincount(y).min() >= 2 else None
            try:
                X, _, y, _ = train_test_split(X, y, train_size=self.max_muestras,
                                              stratify=estrat, random_state=self.semilla)
            except ValueError:
                idx = np.random.default_rng(self.semilla).choice(len(y), self.max_muestras, replace=False)
                # Convertimos a arrays de numpy para permitir indexación múltiple
                X = np.array(X)[idx]
                y = np.array(y)[idx]
 
        ds = DatasetLocal(key=key, X=X, y=y, task_type=task)
        self._cache[key] = ds
        return ds
 
 
def registro_por_defecto() -> RegistroDatasets:
    """Registro de demostración con datasets de scikit-learn."""
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    reg = RegistroDatasets()
    reg.registrar("breast_cancer", lambda: load_breast_cancer(return_X_y=True),
                  aliases=["Breast Cancer Wisconsin", "Wisconsin Breast Cancer"])
    reg.registrar("iris", lambda: load_iris(return_X_y=True), aliases=["Iris flowers"])
    reg.registrar("wine", lambda: load_wine(return_X_y=True), aliases=["Wine recognition"])
    return reg
 
 
# ==============================================================================
# 4. MATEMÁTICAS: GRAM, TARGETS, LINEALIDAD, PCA, ALINEACIÓN, g_CQ, s, d_eff
# ==============================================================================
 
def normalizar_matriz_gram(K: np.ndarray) -> np.ndarray:
    """Simetriza y normaliza K (N x N) para que Tr(K) = N."""
    K = (K + K.T) / 2.0
    traza = np.trace(K)
    if traza <= 0:
        raise ValueError("La traza de la matriz de Gram debe ser estrictamente positiva.")
    return K * (K.shape[0] / traza)
 
 
def inferir_task_type(y: np.ndarray, task_type: str = "auto") -> str:
    if task_type in ("classification", "regression"):
        return task_type
    es_clasif = (not np.issubdtype(y.dtype, np.floating)) or (
        np.all(np.mod(y, 1) == 0) and len(np.unique(y)) <= 10)
    return "classification" if es_clasif else "regression"
 
 
def preparar_target(y: np.ndarray, task_type: str) -> np.ndarray:
    """
    Devuelve T (N x m) simétrico respecto a 0 para s_Q y la alineación:
      binario -> +-1 (m=1) | multiclase -> one-vs-rest +-1 (m=#clases) | regresión -> y estandarizado.
    (Usar etiquetas {0,1} crudas sesga s_Q y la alineación.)
    """
    if task_type == "classification":
        clases = np.unique(y)
        if len(clases) == 2:
            return np.where(y == clases[1], 1.0, -1.0)[:, None]
        return (y[:, None] == clases[None, :]).astype(float) * 2.0 - 1.0
    y = y.astype(float)
    sd = y.std()
    return ((y - y.mean()) / (sd if sd > 1e-12 else 1.0))[:, None]
 
 
def calcular_delta_linealidad(X: np.ndarray, y: np.ndarray, task_type: str = "auto",
                              cv: int = 5, semilla: int = SEMILLA) -> float:
    """
    delta_lin = 1 - (Score_Lineal / Score_RandomForest), evaluado con VALIDACIÓN CRUZADA
    (evaluar en train, como antes, deja al RF en ~1.0 por sobreajuste y distorsiona delta).
    """
    task = inferir_task_type(y, task_type)
    if task == "classification":
        min_clase = int(np.bincount(y.astype(int)).min()) if np.issubdtype(y.dtype, np.integer) else cv
        splitter = StratifiedKFold(n_splits=max(2, min(cv, min_clase)), shuffle=True, random_state=semilla)
        lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        rf = RandomForestClassifier(n_estimators=100, random_state=semilla, n_jobs=-1)
        scoring = "accuracy"
    else:
        splitter = KFold(n_splits=max(2, min(cv, len(y))), shuffle=True, random_state=semilla)
        lin = make_pipeline(StandardScaler(), LinearRegression())
        rf = RandomForestRegressor(n_estimators=100, random_state=semilla, n_jobs=-1)
        scoring = "r2"
 
    score_lin = max(0.0, float(np.mean(cross_val_score(lin, X, y, cv=splitter, scoring=scoring))))
    score_rf = max(1e-5, float(np.mean(cross_val_score(rf, X, y, cv=splitter, scoring=scoring))))
    return float(np.clip(1.0 - score_lin / score_rf, 0.0, 1.0))
 
 
def calcular_varianza_pca(X: np.ndarray, k_latente: int) -> float:
    """Varianza explicada acumulada por PCA clásico con k componentes."""
    n_components = max(1, min(k_latente, X.shape[1], X.shape[0]))
    pca = PCA(n_components=n_components).fit(X)
    return float(np.sum(pca.explained_variance_ratio_))
 
 
def compute_kernel_target_alignment(K: np.ndarray, y: np.ndarray, task_type: str = "auto") -> float:
    """Alineación centrada entre K y T T^T. Devuelve un valor en [-1, 1]."""
    task = inferir_task_type(y, task_type)
    T = preparar_target(y, task)
    K_norm = normalizar_matriz_gram(K)
 
    def centrar(M: np.ndarray) -> np.ndarray:        # H M H sin construir H
        return M - M.mean(axis=0, keepdims=True) - M.mean(axis=1, keepdims=True) + M.mean()
 
    K_c = centrar(K_norm)
    Y_c = centrar(T @ T.T)
    den = np.linalg.norm(K_c, "fro") * np.linalg.norm(Y_c, "fro")
    if den <= 1e-10:
        return 0.0
    return float(np.clip(np.sum(K_c * Y_c) / den, -1.0, 1.0))
 
 
def compute_regularized_gCQ(K_Q: np.ndarray, K_C_list: List[np.ndarray],
                            lambda_reg: float = LAMBDA_REG) -> float:
    """g_CQ* = min_C sqrt( || sqrt(K_Q) (K_C + lambda I)^-1 sqrt(K_Q) ||_inf )."""
    N = K_Q.shape[0]
    K_Q_norm = normalizar_matriz_gram(K_Q)
    eigvals, eigvecs = np.linalg.eigh(K_Q_norm)
    sqrt_KQ = (eigvecs * np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T
 
    I_N = np.eye(N)
    g_values = []
    for K_C in K_C_list:
        K_C_norm = normalizar_matriz_gram(K_C)
        inv_times_sqrt = np.linalg.solve(K_C_norm + lambda_reg * I_N, sqrt_KQ)   # más estable que inv()
        M = sqrt_KQ @ inv_times_sqrt
        M = (M + M.T) / 2.0
        g_values.append(np.sqrt(max(0.0, float(np.linalg.eigvalsh(M)[-1]))))
    return float(np.min(g_values))
 
 
def compute_quantum_complexity(K: np.ndarray, y: np.ndarray, lambda_reg: float = LAMBDA_REG,
                               task_type: str = "auto") -> float:
    """s = tr( T^T (K + lambda I)^-1 T ) / m  con T el target simétrico (ver preparar_target)."""
    task = inferir_task_type(y, task_type)
    T = preparar_target(y, task)
    K_norm = normalizar_matriz_gram(K)
    sol = np.linalg.solve(K_norm + lambda_reg * np.eye(K_norm.shape[0]), T)
    return max(0.0, float(np.sum(T * sol) / T.shape[1]))
 
 
def compute_min_classical_complexity(K_C_list: List[np.ndarray], y: np.ndarray,
                                     lambda_reg: float = LAMBDA_REG, task_type: str = "auto") -> float:
    return float(np.min([compute_quantum_complexity(K, y, lambda_reg, task_type) for K in K_C_list]))
 
 
def construir_familia_kernels_clasicos(X: np.ndarray) -> List[np.ndarray]:
    """
    Familia K_C_list: lineal, polinómico (d=2,3) y RBF. Se espera X ESTANDARIZADA; los gamma del RBF
    se escalan por 1/d para que no dependan de la dimensión del dataset.
    """
    d = max(1, X.shape[1])
    K_list = [linear_kernel(X)]
    K_list += [polynomial_kernel(X, degree=deg) for deg in (2, 3)]
    K_list += [rbf_kernel(X, gamma=g / d) for g in (0.01, 0.1, 1.0, 10.0)]
    return [normalizar_matriz_gram(K) for K in K_list]
 
 
def calcular_dimension_efectiva(K_Q: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(normalizar_matriz_gram(K_Q))
    eig = eig[eig > 1e-10]
    if eig.size == 0:
        return 0.0
    return float(np.sum(eig) ** 2 / np.sum(eig ** 2))
 
 
# ---- Métricas por niveles ----------------------------------------------------
 
def metricas_nivel_dataset(ds: DatasetLocal, lambda_reg: float = LAMBDA_REG) -> Dict[str, float]:
    """Solo dependen de (X, y): se calculan UNA vez por dataset y se cachean."""
    if ds.base is None:
        ds.X_std = StandardScaler().fit_transform(ds.X)
        ds.K_C_list = construir_familia_kernels_clasicos(ds.X_std)
        ds.base = {
            "n_samples": int(len(ds.y)),
            "n_features_original": int(ds.X.shape[1]),
            "n_classes": ds.n_classes,
            "delta_linearity": calcular_delta_linealidad(ds.X, ds.y, ds.task_type),
            "s_C_min": compute_min_classical_complexity(ds.K_C_list, ds.y, lambda_reg, ds.task_type),
        }
    return ds.base
 
 
def metricas_nivel_experimento(ds: DatasetLocal, K_Q: np.ndarray, k_latente: int,
                               lambda_reg: float = LAMBDA_REG) -> Dict[str, float]:
    """Dependen del kernel cuántico K_Q de ESTE experimento."""
    metricas_nivel_dataset(ds, lambda_reg)               # asegura cachés
    N = len(ds.y)
    if ds.K_C_list is None:
        raise ValueError("ds.K_C_list no puede ser None.")
        
    if ds.X_std is None:
        raise ValueError("ds.X_std no puede ser None. Falló la estandarización previa.")

    g_cq = compute_regularized_gCQ(K_Q, ds.K_C_list, lambda_reg)
    s_q = compute_quantum_complexity(K_Q, ds.y, lambda_reg, ds.task_type)
    
    return {
        "k_latente": int(k_latente),
        "eta_pca_k": calcular_varianza_pca(ds.X_std, k_latente),
        "g_CQ_star": g_cq,
        "s_Q": s_q,
        "psi_huang": float(g_cq * np.sqrt(s_q / N)),
        "A_c_Q": compute_kernel_target_alignment(K_Q, ds.y, ds.task_type),
        "d_eff_Q": calcular_dimension_efectiva(K_Q),
    }
 
 
def calcular_metricas_matematicas(
    X_train: np.ndarray, y_train: np.ndarray, K_Q: np.ndarray, k_latente: int,
    K_C_list: Optional[List[np.ndarray]] = None, task_type: str = "auto",
) -> Dict[str, float]:
    """Integradora (compatibilidad con la versión anterior): dataset + experimento en un paso."""
    task = inferir_task_type(np.asarray(y_train), task_type)
    
    # Envolvemos en np.asarray para asegurar que Pylance reconozca 'y' como ndarray
    if task == "classification":
        y = np.asarray(LabelEncoder().fit_transform(y_train))
    else:
        y = np.asarray(y_train, dtype=float)
        
    ds = DatasetLocal(key="adhoc", X=np.asarray(X_train, float), y=y, task_type=task)
    
    if K_C_list is not None:                              # kernels clásicos proporcionados por el usuario
        ds.X_std = StandardScaler().fit_transform(ds.X)
        ds.K_C_list = [normalizar_matriz_gram(K) for K in K_C_list]
        ds.base = {
            "n_samples": int(y.shape[0]), 
            "n_features_original": int(ds.X.shape[1]), 
            "n_classes": ds.n_classes,
            "delta_linearity": calcular_delta_linealidad(ds.X, y, task),
            "s_C_min": compute_min_classical_complexity(ds.K_C_list, y, LAMBDA_REG, task),
        }
        
    out = dict(metricas_nivel_dataset(ds))
    out.update(metricas_nivel_experimento(ds, K_Q, k_latente))
    return out
 
# ==============================================================================
# 5. KERNEL CUÁNTICO: simulador estadovector (reemplazable por tu propio proveedor)
# ==============================================================================
# Un "kernel_provider" es un callable  (X_std, registro_experimento) -> K_Q
# (o (K_Q, fuente) o (K_Q, fuente, n_qubits_simulados)). Sustitúyelo por el kernel REAL
# del paper (Qiskit, PennyLane...) si dispones de él; el simulador es una aproximación.
 
def _hadamard_todos(psi: np.ndarray, n: int) -> np.ndarray:
    N = psi.shape[0]
    psi = psi.reshape((N,) + (2,) * n)
    for ax in range(1, n + 1):
        a, b = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
        psi = np.stack([a + b, a - b], axis=ax) / np.sqrt(2.0)
    return psi.reshape(N, -1)
 
 
def _estados_angle(Xa: np.ndarray) -> np.ndarray:
    psi = np.ones((Xa.shape[0], 1), dtype=complex)
    for q in range(Xa.shape[1]):
        single = np.stack([np.cos(Xa[:, q] / 2), np.sin(Xa[:, q] / 2)], axis=1)
        psi = (psi[:, :, None] * single[:, None, :]).reshape(Xa.shape[0], -1)
    return psi
 
 
def _estados_zz(Xa: np.ndarray, reps: int) -> np.ndarray:
    """ZZFeatureMap (entrelazamiento lineal completo): H^n y fases diagonales, repetido 'reps' veces."""
    N, n = Xa.shape
    dim = 2 ** n
    idx = np.arange(dim)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)   # (dim, n)
    psi = np.zeros((N, dim), dtype=complex)
    psi[:, 0] = 1.0
    for _ in range(max(1, reps)):
        psi = _hadamard_todos(psi, n)
        theta = 2.0 * Xa @ bits.T                                                   # P(2 x_i)
        for i in range(n):
            for j in range(i + 1, n):
                coef = 2.0 * (np.pi - Xa[:, i]) * (np.pi - Xa[:, j])
                theta += coef[:, None] * np.logical_xor(bits[:, i], bits[:, j])[None, :]
        psi = psi * np.exp(1j * theta)
    return psi
 
 
def kernel_provider_simulado(X_std: np.ndarray, reg: Dict[str, Any]) -> Tuple[np.ndarray, str, int]:
    """Kernel de fidelidad |<psi_i|psi_j>|^2 simulado según encoding_type / n_qubits / circuit_depth."""
    n_q = int(min(reg["n_qubits"], MAX_QUBITS_SIM, X_std.shape[1]))
    n_q = max(1, n_q)
    Xr = PCA(n_components=n_q).fit_transform(X_std) if X_std.shape[1] > n_q else X_std
    Xa = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(Xr)  # type: ignore
 
    enc = (reg.get("encoding_type") or "").lower()
    if "angle" in enc or enc in ("ry", "rotation") or "basis" in enc:
        psi, fuente = _estados_angle(Xa), "sim_angle"
    elif "zz" in enc or "iqp" in enc:
        psi, fuente = _estados_zz(Xa, int(reg.get("circuit_depth") or 1)), "sim_zz"
    else:
        psi, fuente = _estados_zz(Xa, int(reg.get("circuit_depth") or 1)), "sim_zz_fallback"
    G = psi @ psi.conj().T
    return np.abs(G) ** 2, fuente, n_q
 
 
def _obtener_kernel(provider: Callable, X_std: np.ndarray, reg: Dict[str, Any]) -> Tuple[np.ndarray, str, int]:
    salida = provider(X_std, reg)
    if isinstance(salida, tuple):
        K = salida[0]
        fuente = salida[1] if len(salida) > 1 else "custom"
        nq = salida[2] if len(salida) > 2 else reg.get("n_qubits")
    else:
        K, fuente, nq = salida, "custom", reg.get("n_qubits")
        
    # Usamos (nq or 0) para que si es None, pase a ser 0 antes de convertirse a int
    return np.asarray(K, dtype=float), str(fuente), int(nq or 0)
 
 
# ==============================================================================
# 6. FUSIÓN: columnas del paper + columnas calculadas  ->  fila completa
# ==============================================================================
 
def _valor_valido(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (float, np.floating)):
        return bool(np.isfinite(v))
    if isinstance(v, str):
        return v.strip() != ""
    return True
 
 
def completar_y_validar_fila(fila: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Aplica defaults trazables, deriva campos y comprueba que no falte NINGUNA columna."""
    fila = dict(fila)

    # defaults / derivados con trazabilidad
    fila["is_real_hardware"] = bool(fila.get("is_real_hardware") or False)
    fila["kernel_concentration_observed"] = bool(fila.get("kernel_concentration_observed") or False)

    fila["circuit_depth_imputed"] = fila.get("circuit_depth") is None
    if fila["circuit_depth_imputed"]:
        fila["circuit_depth"] = 1

    sc, sq = fila.get("score_classical"), fila.get("score_quantum")
    # Añadimos 'is not None' para Pylance, y aseguramos la validación de valor
    if sc is not None and sq is not None and _valor_valido(sc) and _valor_valido(sq):
        # Convertimos explícitamente a float para operaciones matemáticas seguras
        sc_f = float(sc)
        sq_f = float(sq)
        
        menor_mejor = _es_metrica_de_error(fila.get("metric_name"))
        fila["score_gap"] = float((sc_f - sq_f) if menor_mejor else (sq_f - sc_f))  # >0 => ventaja cuántica
        
        fila["advantage_derived"] = fila.get("quantum_advantage_reported") is None
        if fila["advantage_derived"]:
            fila["quantum_advantage_reported"] = bool(fila["score_gap"] > 0.01)

    for col in ("experiment_label", "quantum_model", "classical_baseline", "encoding_type", "metric_name"):
        if not _valor_valido(fila.get(col)):
            fila[col] = "unspecified"

    faltan = [c for c in COLUMNAS_FINALES if not _valor_valido(fila.get(c))]
    if faltan:
        return None, f"columnas vacías: {faltan}"
    
    return {c: fila[c] for c in COLUMNAS_FINALES}, None
 
 
def enriquecer_registro(
    reg: Dict[str, Any],
    registro_ds: RegistroDatasets,
    kernel_provider: Callable = kernel_provider_simulado,
    lambda_reg: float = LAMBDA_REG,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Convierte un registro de la API (1 experimento) en una fila completa. Devuelve (fila, motivo_fallo)."""
    reg = dict(reg)
    reg["score_classical"] = _normalizar_score(reg.get("score_classical"), reg.get("metric_name"))
    reg["score_quantum"] = _normalizar_score(reg.get("score_quantum"), reg.get("metric_name"))
 
    for obligatorio in ("n_qubits", "score_classical", "score_quantum"):
        if not _valor_valido(reg.get(obligatorio)):
            return None, f"el paper no reporta '{obligatorio}' para este experimento"
    if int(reg["n_qubits"]) < 1:
        return None, "n_qubits inválido"
 
    key = registro_ds.resolver(reg.get("local_dataset_key_sugerida"), reg.get("dataset_name"))
    if key is None:
        return None, f"sin datos locales para el dataset '{reg.get('dataset_name')}' (regístralo en RegistroDatasets)"
 
    try:
        ds = registro_ds.obtener(key)
        if reg.get("task_type_paper") and reg["task_type_paper"].lower() != ds.task_type:
            logger.warning("Dataset '%s': el paper dice '%s' pero los datos locales parecen '%s'. Se usa el local.",
                           key, reg["task_type_paper"], ds.task_type)

        base = metricas_nivel_dataset(ds, lambda_reg)                       # cacheado por dataset
        
        # --- NUEVA VALIDACIÓN AQUÍ ---
        if ds.X_std is None:
            raise ValueError(f"ds.X_std no está inicializado para el dataset {key}")
            
        K_Q, fuente, n_q_sim = _obtener_kernel(kernel_provider, ds.X_std, reg)
        k_latente = max(1, min(int(reg["n_qubits"]), ds.X.shape[1]))
        exp_m = metricas_nivel_experimento(ds, K_Q, k_latente, lambda_reg)
    except Exception as e:  # LinAlgError, memoria, datos mal formados...
        return None, f"error calculando métricas: {type(e).__name__}: {e}"

        
    stem = Path(reg["archivo_origen"]).stem
    fila: Dict[str, Any] = {
        "experiment_id": f"{stem}|{key}|d{reg['dataset_idx']}e{reg['experiment_idx']}",
        "archivo_origen": reg["archivo_origen"],
        "paper_title": reg.get("paper_title") or stem,
        "dataset_name": reg["dataset_name"],
        "dataset_local_key": key,
        "domain_type": reg.get("domain_type"),
        "task_type": ds.task_type,
        "n_qubits_simulados": n_q_sim,
        "kernel_source": fuente,
    }
    for c in ("experiment_label", "quantum_model", "classical_baseline", "encoding_type", "n_qubits",
              "circuit_depth", "is_real_hardware", "metric_name", "score_classical", "score_quantum",
              "quantum_advantage_reported", "kernel_concentration_observed"):
        fila[c] = reg.get(c)
    fila.update(base)
    fila.update(exp_m)
    return completar_y_validar_fila(fila)
 
 
# ==============================================================================
# 7. ESCRITURA SEGURA DEL CSV
# ==============================================================================
 
def guardar_filas_en_csv(
    filas: List[Dict[str, Any]],
    output_csv_path: str,
    columnas: Optional[List[str]] = COLUMNAS_FINALES,
    reemplazar_archivos: Iterable[str] = (),
):
    """
    Anexa filas manteniendo SIEMPRE el mismo esquema de columnas (antes, anexar con columnas
    distintas desalineaba el CSV). Reglas:
      - Reindexa al esquema fijo (o a la unión de columnas si columnas=None).
      - Re-procesar un paper reemplaza sus filas anteriores (reemplazar_archivos).
      - Deduplica por experiment_id si existe.
      - Escritura atómica (archivo temporal + os.replace).
    """
    if not filas:
        logger.info("No hay filas que guardar en '%s'.", output_csv_path)
        return
 
    destino = Path(output_csv_path)
    df_new = pd.DataFrame(filas)
    df = df_new
    if destino.exists() and destino.stat().st_size > 0:
        df_old = pd.read_csv(destino)
        reemplazar = set(reemplazar_archivos)
        if reemplazar and "archivo_origen" in df_old.columns:
            df_old = df_old[~df_old["archivo_origen"].isin(reemplazar)]
        df = pd.concat([df_old, df_new], ignore_index=True)
 
    if "experiment_id" in df.columns:
        df = df.drop_duplicates(subset="experiment_id", keep="last")
 
    if columnas is not None:
        extras = [c for c in df.columns if c not in columnas]
        df = df.reindex(columns=columnas + extras)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, destino)
    logger.info("Guardadas %d filas nuevas en '%s' (total: %d).", len(df_new), output_csv_path, len(df))
 
 
def _ruta_pendientes(output_csv_path: str) -> str:
    p = Path(output_csv_path)
    return str(p.with_name(p.stem + "_pendientes" + p.suffix))
 
 
# ==============================================================================
# 8. FLUJOS PÚBLICOS
# ==============================================================================
 
def procesar_paper_pdf(
    pdf_path: str,
    registro_ds: RegistroDatasets,
    client_api: Optional[genai.Client] = None,
    kernel_provider: Callable = kernel_provider_simulado,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """PDF -> (filas_completas, filas_pendientes). Una fila por experimento de cada dataset del paper."""
    nombre = Path(pdf_path).name
    reporte = extraer_reporte_de_pdf(pdf_path, client_api, datasets_disponibles=registro_ds.claves())
    if reporte is None:
        return [], [{"archivo_origen": nombre, "motivo": "fallo en la extracción con Gemini"}]
 
    registros = aplanar_reporte(reporte, nombre)
    logger.info("'%s': %d datasets, %d experimentos extraídos.", nombre, len(reporte.datasets), len(registros))
 
    completas, pendientes = [], []
    for reg in registros:
        fila, motivo = enriquecer_registro(reg, registro_ds, kernel_provider)
        if fila is not None:
            completas.append(fila)
        else:
            pendientes.append({**reg, "motivo": motivo})
            logger.warning("Pendiente [%s | %s]: %s", reg["dataset_name"], reg.get("experiment_label"), motivo)
    return completas, pendientes
 
 
def registrar_paper_pdf_en_csv(
    pdf_path: str,
    output_csv_path: str,
    registro_ds: Optional[RegistroDatasets] = None,
    client_api: Optional[genai.Client] = None,
    kernel_provider: Callable = kernel_provider_simulado,
):
    """MODO 1: paper PDF -> N datasets x M experimentos -> filas COMPLETAS en el CSV."""
    registro_ds = registro_ds or registro_por_defecto()
    completas, pendientes = procesar_paper_pdf(pdf_path, registro_ds, client_api, kernel_provider)
    nombre = Path(pdf_path).name
    guardar_filas_en_csv(completas, output_csv_path, reemplazar_archivos=[nombre])
    if pendientes:
        guardar_filas_en_csv(pendientes, _ruta_pendientes(output_csv_path), columnas=None,
                             reemplazar_archivos=[nombre])
    logger.info("'%s': %d filas completas, %d pendientes.", nombre, len(completas), len(pendientes))
 
 
def registrar_experimento_local_en_csv(
    output_csv_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    K_Q: np.ndarray,
    k_latente: int = 2,
    metadatos_experimento: Optional[Dict[str, Any]] = None,
):
    """MODO 2: experimento propio (X, y, K_Q) + metadatos -> fila completa en el mismo esquema."""
    meta = dict(metadatos_experimento or {})
    metricas = calcular_metricas_matematicas(X_train, y_train, K_Q, k_latente)
 
    fila: Dict[str, Any] = {**meta, **metricas}
    fila.setdefault("archivo_origen", "experimento_local")
    fila.setdefault("paper_title", "local")
    fila.setdefault("dataset_local_key", meta.get("dataset_name", "local"))
    fila.setdefault("task_type", inferir_task_type(np.asarray(y_train)))
    fila.setdefault("n_qubits_simulados", meta.get("n_qubits"))
    fila.setdefault("kernel_source", "custom")
    fila.setdefault("experiment_label", f"{meta.get('encoding_type', 'local')} local")
    fila.setdefault("experiment_id", f"local|{meta.get('dataset_name', 'ds')}|{meta.get('encoding_type', '')}|"
                                     f"{meta.get('n_qubits', '')}|{meta.get('circuit_depth', '')}")
 
    fila_ok, motivo = completar_y_validar_fila(fila)
    if fila_ok is None:
        logger.error("Experimento local NO guardado: %s", motivo)
        guardar_filas_en_csv([{**fila, "motivo": motivo}], _ruta_pendientes(output_csv_path), columnas=None)
        return
    guardar_filas_en_csv([fila_ok], output_csv_path)
 
 
# ==============================================================================
# 9. EJEMPLO DE EJECUCIÓN (MAIN) — al final del archivo para que todo esté definido
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    CSV_META_DATASET = "meta_dataset_ventaja_cuantica.csv"
 
    # Registra aquí los datasets que aparecen en tus papers (nombre del paper -> datos locales)
    registro = registro_por_defecto()
    # registro.registrar_csv("german_credit", "data/german.csv", target_col="class",
    #                        aliases=["German Credit", "Statlog German Credit"])
 
    # CASO A: todos los PDFs de la carpeta 'papers/'
    for pdf in sorted(Path("papers").glob("*.pdf")):
        print(f"\n--- PROCESANDO PAPER: {pdf.name} ---")
        registrar_paper_pdf_en_csv(str(pdf), CSV_META_DATASET, registro_ds=registro)
 
    # CASO B: experimento local con datos propios
    print("\n--- PROCESANDO EXPERIMENTO LOCAL ---")
    rng = np.random.default_rng(SEMILLA)
    N = 50
    X_dummy = rng.standard_normal((N, 6))
    y_dummy = rng.choice([0, 1], size=N)
    dist = np.linalg.norm(X_dummy[:, None] - X_dummy[None, :], axis=-1) ** 2
    K_Q_dummy = np.exp(-0.15 * dist)
 
    registrar_experimento_local_en_csv(
        output_csv_path=CSV_META_DATASET,
        X_train=X_dummy, y_train=y_dummy, K_Q=K_Q_dummy, k_latente=3,
        metadatos_experimento={
            "dataset_name": "Synthetic_Bloch_Data", "domain_type": "synthetic",
            "quantum_model": "QSVM", "classical_baseline": "SVM-RBF", "metric_name": "accuracy",
            "encoding_type": "ZZFeatureMap", "n_qubits": 6, "circuit_depth": 2,
            "is_real_hardware": False, "kernel_concentration_observed": False,
            "score_classical": 0.74, "score_quantum": 0.91, "quantum_advantage_reported": True,
        },
    )
 