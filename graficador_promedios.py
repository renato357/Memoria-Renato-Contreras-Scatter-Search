import json
import matplotlib.pyplot as plt
import os

def graficar_data_leakage():
    # Archivos exactos del Test A/B
    archivos = {
        "Validación Estándar (Con Leakage)": {
            "ruta": "resultado_eval_estandar.json", 
            "color": "#1f77b4", # Azul profesional
            "marcador": "o-"
        },
        "Validación Cruzada (Excluyente)": {
            "ruta": "resultado_eval_cruzada.json", 
            "color": "#d62728", # Rojo para destacar la exclusión
            "marcador": "s-"
        }
    }

    plt.figure(figsize=(10, 6))

    for etiqueta, config in archivos.items():
        ruta = config["ruta"]
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extraemos las generaciones y la métrica exacta ("mejor_sbert")
                    generaciones = [x['generacion'] for x in data['historial_convergencia']]
                    mejor_sbert = [x['mejor_sbert'] for x in data['historial_convergencia']]
                    
                    plt.plot(generaciones, mejor_sbert, config["marcador"], color=config["color"], 
                             label=etiqueta, linewidth=2.5, markersize=8)
            except Exception as e:
                print(f"Error cargando {ruta}: {e}")
        else:
            print(f"[Advertencia] No se encontró el archivo '{ruta}'.")

    # Configuración visual optimizada para diapositivas
    plt.title('Impacto del Data Leakage en la Convergencia SBERT', fontsize=15, fontweight='bold')
    plt.xlabel('Generación', fontsize=13)
    plt.ylabel('SBERT Máximo (Calidad Semántica)', fontsize=13)
    
    # Fuerzo a que el eje X muestre todas las generaciones (0 a 10)
    plt.xticks(range(0, 11))
    
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    nombre_salida = 'curva_data_leakage.png'
    plt.savefig(nombre_salida, dpi=300)
    print(f"¡Éxito! Gráfico del Test A/B guardado como: {nombre_salida}")

if __name__ == "__main__":
    print("=== GENERADOR DE GRÁFICO: DATA LEAKAGE ===")
    graficar_data_leakage()
    plt.show()