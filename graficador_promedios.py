import json
import matplotlib.pyplot as plt
import numpy as np

def promediar_historial(archivos):
    """Lee una lista de archivos JSON y retorna el promedio de max_sbert por generación"""
    historiales = []
    
    for ruta in archivos:
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                data = json.load(f)
                max_sbert = [x['max_sbert'] for x in data['historial_convergencia']]
                historiales.append(max_sbert)
        except Exception as e:
            print(f"Error cargando {ruta}: {e}")
            
    if not historiales:
        return []
        
    matriz = np.array(historiales)
    promedios = np.mean(matriz, axis=0)
    return promedios.tolist()

def graficar_convergencia_promediada():
    archivos_10x10 = [
        "resultado_init_10x10.json",  
        "resultado_init_10x10_2.json",
        "resultado_init_10x10_3.json",
        "resultado_init_10x10_4.json",
        "resultado_init_10x10_5.json"
    ]
    
    archivos_llm = [
        "resultado_init_llm.json",   
        "resultado_init_llm_2.json", 
        "resultado_init_llm_3.json",
        "resultado_init_llm_4.json",
        "resultado_init_llm_5.json"
    ]

    plt.figure(figsize=(10, 6))

    # Sacamos los promedios
    promedios_10x10 = promediar_historial(archivos_10x10)
    promedios_llm = promediar_historial(archivos_llm)
    
    # Generaciones del 1 al 10
    generaciones = list(range(1, 11))
    
    # Graficamos las líneas promediadas
    if promedios_10x10:
        plt.plot(generaciones, promedios_10x10, 'r--^', label="Init 10x10 (Promedio 5 runs)", linewidth=2, markersize=6)
    if promedios_llm:
        plt.plot(generaciones, promedios_llm, 'b-o', label="Init LLM + Cruce Puro (Promedio 5 runs)", linewidth=2, markersize=6)

    # Configuración estética
    plt.title('Convergencia Promedio de SBERT (5 Ejecuciones Independientes)', fontsize=14, fontweight='bold')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('SBERT Máximo Promedio', fontsize=12)
    plt.xticks(range(1, 11))
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    nombre_salida = 'curva_promediada_final.png'
    plt.savefig(nombre_salida, dpi=300, bbox_inches='tight')
    print(f"\n¡Éxito! Gráfico promediado guardado como: {nombre_salida}")
    plt.show()

if __name__ == "__main__":
    graficar_convergencia_promediada()