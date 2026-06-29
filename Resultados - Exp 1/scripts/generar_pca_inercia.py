import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def generar_pca_e_inercia():
    ruta_json = os.path.join(os.path.dirname(__file__), '..', 'exp1_base.json')
    ruta_out = os.path.join(os.path.dirname(__file__), '..', 'pca_niching.png')
    
    with open(ruta_json, "r", encoding="utf-8") as f:
        runs = json.load(f)
        
    print("Cargando modelo SBERT para PCA e Inercia...")
    modelo_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. CALCULAR INERCIA PROMEDIO DE LAS 5 CORRIDAS
    inercias_iniciales = []
    inercias_finales = []
    
    for run in runs:
        textos_ini = [ind.get('tweet_generado', '') for ind in run.get('poblacion_inicial', [])]
        textos_fin = [ind.get('tweet_generado', '') for ind in run.get('refset_final', [])]
        
        # Filtrar vacíos
        textos_ini = [t for t in textos_ini if t.strip()]
        textos_fin = [t for t in textos_fin if t.strip()]
        
        emb_ini = modelo_sbert.encode(textos_ini)
        emb_fin = modelo_sbert.encode(textos_fin)
        
        if len(emb_ini) >= 3:
            kmeans_ini = KMeans(n_clusters=3, n_init=10, random_state=42).fit(emb_ini)
            inercias_iniciales.append(kmeans_ini.inertia_)
        if len(emb_fin) >= 3:
            kmeans_fin = KMeans(n_clusters=3, n_init=10, random_state=42).fit(emb_fin)
            inercias_finales.append(kmeans_fin.inertia_)
            
    print("\n=== ANÁLISIS DE DIVERSIDAD (K-MEANS INERCIA, N=5) ===")
    print(f"Inercia Promedio Población Inicial: {np.mean(inercias_iniciales):.4f}")
    print(f"Inercia Promedio RefSet Final:      {np.mean(inercias_finales):.4f}")
    
    # 2. GENERAR GRÁFICO PCA DE LA PRIMERA CORRIDA (Representativo)
    print("\nGenerando Gráfico PCA para la primera corrida...")
    run = runs[0]
    textos_iniciales = [ind['tweet_generado'] for ind in run['poblacion_inicial']]
    textos_finales = [ind['tweet_generado'] for ind in run['refset_final']]
    textos_reales = [ind['tweet_real_match'] for ind in run['refset_final']]
    
    todos_textos = textos_iniciales + textos_finales + textos_reales
    embeddings = modelo_sbert.encode(todos_textos)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    coords_ini = coords[:len(textos_iniciales)]
    coords_fin = coords[len(textos_iniciales):len(textos_iniciales)+len(textos_finales)]
    coords_reales = coords[len(textos_iniciales)+len(textos_finales):]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_ini[:, 0], coords_ini[:, 1], c='blue', alpha=0.3, label='Población Inicial', s=50)
    plt.scatter(coords_fin[:, 0], coords_fin[:, 1], c='red', marker='*', label='RefSet Final (Niching)', s=200, edgecolors='black')
    plt.scatter(coords_reales[:, 0], coords_reales[:, 1], c='green', marker='^', label='Target Tweets (Reales)', s=150)
    
    plt.title("Visualización PCA: Efecto del Niching en el Espacio Semántico")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {ruta_out}")

if __name__ == "__main__":
    generar_pca_e_inercia()
