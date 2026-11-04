import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import numpy as np
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


# ==============================================================================
# API COLUMN EXTRACTION
# ==============================================================================
from pydantic import BaseModel, Field
from typing import Optional

class MetadatosPaper(BaseModel):
    # Identificadores (Gemini los inferirá del contexto del paper)
    dataset_name: str = Field(description="Nombre del dataset (ej: Breast Cancer, MedMNIST, German Credit)")
    domain_type: str = Field(description="Dominio del dataset (ej: medical_image, tabular_health, physics, credit)")
    task_type: str = Field(description="Tipo de tarea inferida: 'classification', 'regression' o 'clustering'")
    
    # Configuración del Experimento (Suelen estar en todos los papers)
    encoding_type: Optional[str] = Field(default=None, description="Mapa de características (ej: ZZFeatureMap, Angle, Covariant, PQK)")
    n_qubits: Optional[int] = Field(default=None, description="Número de qubits utilizados")
    circuit_depth: Optional[int] = Field(default=None, description="Profundidad del circuito")
    is_real_hardware: Optional[bool] = Field(default=None, description="True si usaron QPU real, False si usaron simulador")
    
    # Resultados Empíricos (NÚCLEO de los papers de Benchmarking 2023-2026)
    score_classical: Optional[float] = Field(default=None, description="Rendimiento del mejor modelo clásico reportado (Accuracy/F1/AUC)")
    score_quantum: Optional[float] = Field(default=None, description="Rendimiento del modelo cuántico reportado")
    quantum_advantage_reported: Optional[bool] = Field(default=None, description="¿Los autores concluyen que hubo ventaja cuántica significativa?")
    
    # Diagnósticos Teóricos Avanzados (Solo estarán en papers de teoría como Huang 2021)
    g_CQ_star: Optional[float] = Field(default=None, description="Diferencia geométrica g_CQ si el paper la calcula")
    s_Q: Optional[float] = Field(default=None, description="Complejidad cuántica s_Q si el paper la calcula")
    psi_huang: Optional[float] = Field(default=None, description="Cota de ventaja Psi si el paper la menciona")
    kernel_concentration_observed: Optional[bool] = Field(default=None, description="¿Mencionan problemas de concentración del kernel/barren plateaus?")


def extraer_metadatos_de_pdf(pdf_path: str, client_api: Optional[genai.Client] = None) -> Dict[str, Any]:
    """
    Lee un archivo PDF de investigación y extrae sus metadatos estructurados mediante Gemini API.
    
    :param pdf_path: Ruta al archivo PDF del paper.
    :param client_api: Instancia opcional del cliente genai. Si es None, se crea una automáticamente.
    :return: Diccionario con las columnas de metadatos extraídas.
    """
    path = Path(pdf_path)
    
    # Verificación de existencia del archivo
    if not path.is_file():
        raise FileNotFoundError(f"❌ El archivo no existe en la ruta especificada: {pdf_path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"❌ El archivo proporcionado debe ser un PDF. Recibido: {path.suffix}")

    # Inicialización del cliente si no se proporciona uno
    if client_api is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ No se encontró la variable GEMINI_API_KEY en el entorno o archivo .env")
        client_api = genai.Client(api_key=api_key)

    print(f"📄 Procesando paper: {path.name}...")

    # Lectura del archivo en binario
    with open(path, "rb") as f:
        pdf_bytes = f.read()

    prompt = """
    Analiza rigurosamente este artículo de investigación cuántica.
    Extrae la información sobre el conjunto de datos y el experimento cuántico realizado.
    Rellena los campos del esquema JSON proporcionado.
    Si algún dato numérico o métrica no figura de forma explícita en el documento, asigna el valor null.
    """

    try:
        response = client_api.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MetadatosPaper,
                temperature=0.1
            ),
        )
        
        # Asegurarnos de que response.text contenga un string válido
        if not response.text:
            raise ValueError(f"La API de Gemini no devolvió texto en la respuesta para {path.name}.")

        # Validación con Pydantic y conversión a diccionario
        metadatos = MetadatosPaper.model_validate_json(response.text).model_dump()
        metadatos["archivo_origen"] = path.name
        return metadatos

    except Exception as e:
        print(f"❌ Error al procesar {path.name} con la API de Gemini: {str(e)}")
        return {}




# ==============================================================================
# COLUMN CALCULATIONS
# ==============================================================================
import numpy as np
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


# ==============================================================================
# 1. NORMALIZACIÓN DE MATRICES DE GRAM
# ==============================================================================

def normalizar_matriz_gram(K: np.ndarray) -> np.ndarray:
    """
    Normaliza una matriz de Gram K (N x N) para garantizar que Tr(K) = N.
    Condición imprescindible para comparar kernels a la misma escala.
    """
    N = K.shape[0]
    traza = np.trace(K)
    if traza <= 0:
        raise ValueError("La traza de la matriz de Gram debe ser estrictamente positiva.")
    return K * (N / traza)


# ==============================================================================
# 2. LINEALIDAD Y VARIANZA EXPLICADA CLÁSICA
# ==============================================================================

def calcular_delta_linealidad(X: np.ndarray, y: np.ndarray, task_type: str = "auto") -> float:
    """
    Calcula el factor de no linealidad delta_lin = 1 - (Score_Lineal / Score_RandomForest).
    """
    if task_type == "auto":
        is_classification = len(np.unique(y)) <= 10 and not np.issubdtype(y.dtype, np.floating)
    else:
        is_classification = (task_type == "classification")

    if is_classification:
        model_lin = LogisticRegression(max_iter=1000).fit(X, y)
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    else:
        model_lin = LinearRegression().fit(X, y)
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

    score_lin = max(0.0, float(model_lin.score(X, y)))
    score_rf = max(1e-5, float(model_rf.score(X, y)))

    delta_lin = 1.0 - (score_lin / score_rf)
    return float(np.clip(delta_lin, 0.0, 1.0))


def calcular_varianza_pca(X: np.ndarray, k_latente: int) -> float:
    """
    Calcula la varianza explicada acumulada por PCA clásico en k componentes.
    """
    n_components = min(k_latente, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components).fit(X)
    return float(np.sum(pca.explained_variance_ratio_))


# ==============================================================================
# 3. ALINEACIÓN KERNEL-TARGET CENTRADA (A_c)
# ==============================================================================

def compute_kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula la alineación centrada entre la matriz de Gram K y la matriz del 
    producto externo de etiquetas (y * y^T). Retorna un valor en [-1, 1].
    """
    N = len(y)
    K_norm = normalizar_matriz_gram(K)
    
    # Matriz de centrado H = I - (1/N) * 11^T
    H = np.eye(N) - np.ones((N, N)) / N
    K_c = H @ K_norm @ H
    
    # Matriz de etiquetas centrada Y_c = H @ (y y^T) @ H
    y_vec = y.reshape(-1, 1).astype(float)
    Y = y_vec @ y_vec.T
    Y_c = H @ Y @ H
    
    # Producto interno de Frobenius / producto de normas de Frobenius
    num = np.sum(K_c * Y_c)
    den = np.linalg.norm(K_c, 'fro') * np.linalg.norm(Y_c, 'fro')
    
    if den <= 1e-10:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


# ==============================================================================
# 4. GEOMETRÍA Y COMPLEJIDAD CUÁNTICA VS CLÁSICA
# ==============================================================================

def compute_regularized_gCQ(K_Q: np.ndarray, K_C_list: list[np.ndarray], lambda_reg: float = 1e-3) -> float:
    """
    Calcula la diferencia geométrica regularizada mínima g_CQ* frente a una familia de kernels clásicos.
    """
    N = K_Q.shape[0]
    K_Q_norm = normalizar_matriz_gram(K_Q)

    # Raíz cuadrada de K_Q mediante descomposición espectral
    eigvals_Q, eigvecs_Q = np.linalg.eigh(K_Q_norm)
    eigvals_Q = np.maximum(eigvals_Q, 0.0)
    sqrt_KQ = eigvecs_Q @ np.diag(np.sqrt(eigvals_Q)) @ eigvecs_Q.T

    g_values = []
    I_N = np.eye(N)

    for K_C in K_C_list:
        K_C_norm = normalizar_matriz_gram(K_C)
        KC_reg_inv = np.linalg.inv(K_C_norm + lambda_reg * I_N)
        M = sqrt_KQ @ KC_reg_inv @ sqrt_KQ
        
        max_eigval = float(np.max(np.linalg.eigvalsh(M)))
        g_val = np.sqrt(max(0.0, max_eigval))
        g_values.append(g_val)

    return float(np.min(g_values))


def compute_quantum_complexity(K: np.ndarray, y: np.ndarray, lambda_reg: float = 1e-3) -> float:
    """
    Calcula la complejidad de aprendizaje de un modelo de kernel s = y^T * (K + lambda * I)^-1 * y.
    """
    N = len(y)
    K_norm = normalizar_matriz_gram(K)
    y_vec = y.astype(float)
    
    solucion = np.linalg.solve(K_norm + lambda_reg * np.eye(N), y_vec)
    s_val = float(y_vec.T @ solucion)
    return max(0.0, s_val)


def compute_min_classical_complexity(K_C_list: list[np.ndarray], y: np.ndarray, lambda_reg: float = 1e-3) -> float:
    """
    Calcula la menor complejidad alcanzable por la familia de kernels clásicos (s_C_min).
    """
    s_C_values = [compute_quantum_complexity(K_C, y, lambda_reg=lambda_reg) for K_C in K_C_list]
    return float(np.min(s_C_values))


# ==============================================================================
# 5. BASELINES CLÁSICOS
# ==============================================================================

def construir_familia_kernels_clasicos(X: np.ndarray) -> list[np.ndarray]:
    """
    Construye la familia K_C_list (Lineal, Polinómico d=2,3 y RBF gammas=[0.01, 0.1, 1.0, 10.0]).
    """
    K_C_list = [linear_kernel(X)]
    
    for degree in [2, 3]:
        K_C_list.append(polynomial_kernel(X, degree=degree))
        
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        K_C_list.append(rbf_kernel(X, gamma=gamma))
        
    return [normalizar_matriz_gram(K_C) for K_C in K_C_list]


# ==============================================================================
# 6. FUNCIÓN INTEGRADORA DEFINITIVA
# ==============================================================================

def calcular_metricas_matematicas(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    K_Q: np.ndarray, 
    k_latente: int,
    K_C_list: list[np.ndarray] = None
) -> dict:
    """
    Calcula de forma unificada todas las metacaracterísticas matemáticas y espectrales.
    """
    N = len(y_train)

    # Si no se proporciona K_C_list, se genera automáticamente a partir de X_train
    if K_C_list is None:
        K_C_list = construir_familia_kernels_clasicos(X_train)

    # 1. Métricas de Estructura Clásica
    delta_lin = calcular_delta_linealidad(X_train, y_train)
    eta_pca = calcular_varianza_pca(X_train, k_latente)

    # 2. Diferencia Geométrica y Complejidades
    g_CQ_star = compute_regularized_gCQ(K_Q, K_C_list, lambda_reg=1e-3)
    s_Q = compute_quantum_complexity(K_Q, y_train, lambda_reg=1e-3)
    s_C_min = compute_min_classical_complexity(K_C_list, y_train, lambda_reg=1e-3)

    # 3. Cota Teórica de Ventaja de Huang (Psi)
    psi_huang = g_CQ_star * np.sqrt(s_Q / N)

    # 4. Alineación Kernel-Target Centrada (A_c_Q)
    A_c_Q = compute_kernel_target_alignment(K_Q, y_train)

    # 5. Dimensión Efectiva Espectral (d_eff_Q)
    K_Q_norm = normalizar_matriz_gram(K_Q)
    eigenvals = np.linalg.eigvalsh(K_Q_norm)
    eigenvals = eigenvals[eigenvals > 1e-10]
    d_eff_Q = float((np.sum(eigenvals)**2) / np.sum(eigenvals**2))

    return {
        "delta_linearity": delta_lin,
        "eta_pca_k": eta_pca,
        "g_CQ_star": g_CQ_star,
        "s_Q": s_Q,
        "s_C_min": s_C_min,
        "psi_huang": psi_huang,
        "A_c_Q": A_c_Q,
        "d_eff_Q": d_eff_Q
    }