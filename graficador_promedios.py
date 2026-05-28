import json
import matplotlib.pyplot as plt
import numpy as np
import os

def cargar_historiales(base_name):
    """Busca y carga todos los JSON (original + 4 clones) de un caso específico."""
    historiales = []
    sufijos = ["", "_2", "_3", "_4", "_5"]
    
    for suf in sufijos:
        # Busca el archivo en la misma carpeta o dentro de "Resultados/"
        rutas_posibles = [
            f"{base_name}{suf}.json",
            os.path.join("Resultados", f"{base_name}{suf}.json")
        ]
        
        for ruta in rutas_posibles:
            if os.path.exists(ruta):
                try:
                    with open(ruta, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        max_sbert = [x['max_sbert'] for x in data['historial_convergencia']]
                        historiales.append(max_sbert)
                    break # Si lo encuentra bien, deja de buscar ese sufijo
                except Exception as e:
                    print(f"Error al leer {ruta}: {e}")
                    
    return historiales

def graficar_super_completo():
    # Definimos los 4 casos que generamos
    casos = {
        "Init 10x10": {"base": "resultado_init_10x10", "color": "#d62728", "marker": "^"},     # Rojo
        "Init LLM": {"base": "resultado_init_llm", "color": "#1f77b4", "marker": "o"},         # Azul
        "Cruce Puro": {"base": "resultado_coherencia_pura", "color": "#2ca02c", "marker": "s"},# Verde
        "Cruce LLM": {"base": "resultado_coherencia_llm", "color": "#9467bd", "marker": "D"}   # Morado
    }

    plt.figure(figsize=(13, 7.5))
    generaciones = list(range(1, 11))

    for nombre, config in casos.items():
        historiales = cargar_historiales(config["base"])
        
        if not historiales:
            print(f"⚠️ No se encontraron archivos para {nombre}")
            continue

        # 1. Graficamos las corridas individuales (líneas delgadas y transparentes)
        for i, hist in enumerate(historiales):
            # Solo le ponemos etiqueta a la primera línea individual para no reventar la leyenda
            etiqueta = f"{nombre} (Indiv)" if i == 0 else ""
            plt.plot(generaciones, hist, color=config["color"], alpha=0.25, linewidth=1.2, label=etiqueta)

        # 2. Calculamos y graficamos el promedio (línea gordita y con marcadores)
        if len(historiales) > 0:
            matriz = np.array(historiales)
            promedios = np.mean(matriz, axis=0)
            plt.plot(generaciones, promedios, color=config["color"], alpha=1.0, linewidth=3.8, 
                     marker=config["marker"], markersize=8, label=f"{nombre} (PROMEDIO)")

    # Configuración estética
    plt.title('Convergencia SBERT: Todas las Variaciones y Promedios', fontsize=16, fontweight='bold')
    plt.xlabel('Generación', fontsize=13)
    plt.ylabel('SBERT Máximo', fontsize=13)
    plt.xticks(generaciones)
    
    # Ajustamos la leyenda para que se vea ordenada y fuera del gráfico
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    nombre_salida = 'curva_super_completa.png'
    plt.savefig(nombre_salida, dpi=300)
    print(f"\n¡Éxito! Gráfico guardado como: {nombre_salida}")
    plt.show()

if __name__ == "__main__":
    graficar_super_completo()