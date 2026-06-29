import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generar_convergencia():
    # Leer el JSON de la carpeta superior (ya que estamos en scripts/)
    ruta_json = os.path.join(os.path.dirname(__file__), '..', 'exp1_base.json')
    ruta_out = os.path.join(os.path.dirname(__file__), '..', 'convergencia_sbert.png')
    
    with open(ruta_json, "r", encoding="utf-8") as f:
        runs = json.load(f)
        
    historiales = []
    
    # Extraer historial de las 5 corridas
    for run in runs:
        historial = run.get('historial_convergencia', [])
        # Normalizamos la longitud por si acaso
        sberts = [h['mejor_sbert'] for h in historial]
        historiales.append(sberts)
        
    # Calcular promedio por generación
    # Asumimos que todas tienen 11 generaciones (0 al 10)
    generaciones = range(len(historiales[0]))
    promedios = np.mean(historiales, axis=0)
    desviaciones = np.std(historiales, axis=0)
    
    plt.figure(figsize=(10, 6))
    
    # Graficar las líneas individuales semitransparentes
    for i, hist in enumerate(historiales):
        plt.plot(generaciones, hist, color='gray', alpha=0.3, label='Corrida individual' if i==0 else "")
        
    # Graficar el promedio
    plt.plot(generaciones, promedios, color='blue', linewidth=3, label='SBERT Promedio (N=5)')
    
    # Rellenar desviación estándar
    plt.fill_between(generaciones, promedios - desviaciones, promedios + desviaciones, color='blue', alpha=0.1)
    
    plt.xlabel('Generaciones')
    plt.ylabel('Mejor SBERT Encontrado')
    plt.title('Curva de Convergencia: Evolución Semántica (SS-GrIPS Baseline)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {ruta_out}")

if __name__ == "__main__":
    generar_convergencia()
