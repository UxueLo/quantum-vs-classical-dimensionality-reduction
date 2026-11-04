import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd


class MetadatosPaper(BaseModel):
    dataset_name: str = Field(description="Name of the dataset used in the experiment")
    domain_type: str = Field(description="Dataset domain: 'tabular', 'astrophysics', 'medical', 'chemical', 'image', etc.")
    task_type: str = Field(description="task type: 'classification', 'regression', o 'clustering'")
    encoding_type: str = Field(description="Feature map encoding type: 'AngleEncoding', 'ZZFeatureMap', 'IQP', etc.")
    n_qubits: Optional[int] = Field(default=None, description="nr. qubits used on the circuit")
    circuit_depth: Optional[int] = Field(default=None, description="quantum circuit depth")
    is_real_hardware: bool = Field(default=False, description="True if it was deployed in a real quantum computer, False if it was in a simulator")
    n_shots: Optional[int] = Field(default=None, description="nr. of measures/shots")
    method_used: str = Field(description="reduction/kernel main method: 'QKPCA', 'QAE', 'KPCA', 'PCA', etc.")
    score_classical: Optional[float] = Field(default=None, description="performance of the best classical model(F1, Accuracy, R2)")
    score_quantum: Optional[float] = Field(default=None, description="performance of the best quantum model")


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
        
        # Validación con Pydantic y conversión a diccionario
        metadatos = MetadatosPaper.model_validate_json(response.text).model_dump()
        metadatos["archivo_origen"] = path.name
        return metadatos

    except Exception as e:
        print(f"❌ Error al procesar {path.name} con la API de Gemini: {str(e)}")
        return {}




def calcular_metricas_matematicas(X_train, y_train, K_Q, K_C_list, k_latente):
    N = len(y_train)
    
    # Linealidad clásica
    delta_lin = calcular_delta_linealidad(X_train, y_train)
    eta_pca = calcular_varianza_pca(X_train, k_latente)
    
    # Geometría y Complejidad
    g_CQ_star = compute_regularized_gCQ(K_Q, K_C_list, lambda_reg=1e-3)
    s_Q = compute_quantum_complexity(K_Q, y_train, lambda_reg=1e-3)
    s_C_min = compute_min_classical_complexity(K_C_list, y_train, lambda_reg=1e-3)
    
    # Cota teórica de Huang
    psi_huang = g_CQ_star * np.sqrt(s_Q / N)
    
    # Diagnóstico espectral
    eigenvals = np.linalg.eigvalsh(K_Q)
    eigenvals = eigenvals[eigenvals > 1e-10]
    d_eff_Q = (np.sum(eigenvals)**2) / np.sum(eigenvals**2)
    
    return {
        "delta_linearity": delta_lin,
        "eta_pca_k": eta_pca,
        "g_CQ_star": g_CQ_star,
        "s_Q": s_Q,
        "s_C_min": s_C_min,
        "psi_huang": psi_huang,
        "d_eff_Q": d_eff_Q
    }

