import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generar_curva_sensibilidad():
    ruta_json = os.path.join(os.path.dirname(__file__), '..', 'exp2_sensibilidad.json')
    ruta_out = os.path.join(os.path.dirname(__file__), '..', 'curva_sensibilidad.png')
    
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    configs = []
    sberts = []
    tiempos = []
    
    # Orden lógico de configuraciones (Menor a mayor costo computacional G x R)
    orden = ['conf_G10_R5', 'conf_G5_R10', 'conf_G10_R10', 'conf_G15_R10', 'conf_G10_R15']
    labels_simples = ['G10, R5', 'G5, R10', 'G10, R10', 'G15, R10', 'G10, R15']
    
    for c in orden:
        if c in data:
            runs = data[c]
            # Sacamos el SBERT de la última generación del historial para arreglar el bug anterior
            avg_sbert = np.mean([r['historial_convergencia'][-1]['mejor_sbert'] for r in runs])
            avg_time = np.mean([r['configuracion']['tiempo_ejecucion_minutos'] for r in runs])
            
            idx = orden.index(c)
            configs.append(labels_simples[idx])
            sberts.append(avg_sbert)
            tiempos.append(avg_time)
            
    # Crear gráfico con 2 ejes Y
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Configuración de Hiperparámetros (G=Generaciones, R=RefSet)')
    ax1.set_ylabel('Tiempo Promedio (Minutos)', color=color)
    bars = ax1.bar(configs, tiempos, color=color, alpha=0.5, width=0.4, label='Tiempo (Minutos)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Añadir valores sobre las barras de tiempo
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.0f}m', ha='center', va='bottom', color=color, fontweight='bold')
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Mejor SBERT Promedio', color=color)
    line = ax2.plot(configs, sberts, color=color, marker='o', linewidth=3, markersize=8, label='SBERT Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Añadir valores sobre la línea de SBERT
    for i, txt in enumerate(sberts):
        ax2.annotate(f'{txt:.3f}', (configs[i], sberts[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='darkblue', fontweight='bold')
        
    plt.title('Análisis de Sensibilidad: Calidad Semántica vs Costo Computacional')
    fig.tight_layout()
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {ruta_out}")

if __name__ == "__main__":
    generar_curva_sensibilidad()
