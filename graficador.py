import json
import matplotlib.pyplot as plt

def graficar_convergencia():
    # Diccionario con los archivos y el nombre que queremos en la leyenda
    archivos = {
        "Init 10x10": "resultado_init_10x10.json",
        "Init LLM + Cruce Puro": "resultado_coherencia_pura.json",
        "Init LLM + LLM Corrector": "resultado_coherencia_llm.json"
    }

    plt.figure(figsize=(10, 6))

    # Colores y estilos para diferenciar bien
    estilos = ['r--^', 'b-o', 'g-s']
    
    for (etiqueta, ruta), estilo in zip(archivos.items(), estilos):
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extraemos los datos del historial de convergencia
                generaciones = [x['generacion'] for x in data['historial_convergencia']]
                max_sbert = [x['max_sbert'] for x in data['historial_convergencia']]
                
                # Graficamos la línea
                plt.plot(generaciones, max_sbert, estilo, label=etiqueta, linewidth=2, markersize=6)
        except Exception as e:
            print(f"Error cargando {ruta}: {e}")

    # Configuración estética del gráfico
    plt.title('Convergencia de SBERT (Población inicial y Coherencia)', fontsize=14, fontweight='bold')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('SBERT Máximo (Calidad Semántica)', fontsize=12)
    
    # Aseguramos que el eje X muestre números enteros del 1 al 10
    plt.xticks(range(1, 11))
    
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardamos la imagen en alta calidad
    nombre_salida = 'curva_convergencia_final.png'
    plt.savefig(nombre_salida, dpi=300, bbox_inches='tight')
    print(f"\n¡Éxito! Gráfico guardado como: {nombre_salida}")
    
    # Mostramos el gráfico en pantalla
    plt.show()

if __name__ == "__main__":
    graficar_convergencia()