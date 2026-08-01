import json
import os
import matplotlib.pyplot as plt
import numpy as np

def graficar_tasa_exito_paraphrase():
    print("Analizando tasa de éxito del operador Paraphrase...")
    
    base_dir = os.path.dirname(__file__)
    ruta_exp4 = os.path.join(base_dir, "exp4_temperaturas.json")
    ruta_exp1 = os.path.join(base_dir, "Resultados - Exp 1", "exp1_base.json")
    
    if not (os.path.exists(ruta_exp4) and os.path.exists(ruta_exp1)):
        print("Archivos JSON no encontrados.")
        return
        
    with open(ruta_exp4, "r", encoding="utf-8") as f:
        d4 = json.load(f)
    with open(ruta_exp1, "r", encoding="utf-8") as f:
        runs_base = json.load(f)
        
    # Extraer T=0.2 (primeras 3 corridas)
    intentos_02 = 0
    exitos_02 = 0
    for r in runs_base[:3]:
        stats = r.get("stats_grips", {}).get("grips_paraphrase", {})
        intentos_02 += stats.get("intentos", 0)
        exitos_02 += stats.get("exitos", 0)
        
    tasa_02 = (exitos_02 / intentos_02 * 100) if intentos_02 > 0 else 0
    
    datos_temp = {0.2: tasa_02}
    
    # Extraer el resto (0.0, 0.4, 0.6)
    for t_key in d4.keys():
        temp_val = float(t_key.replace("conf_T", ""))
        runs = d4[t_key]
        intentos = 0
        exitos = 0
        for r in runs:
            stats = r.get("stats_grips", {}).get("grips_paraphrase", {})
            intentos += stats.get("intentos", 0)
            exitos += stats.get("exitos", 0)
            
        tasa = (exitos / intentos * 100) if intentos > 0 else 0
        datos_temp[temp_val] = tasa
        
    temps_ordenadas = sorted(datos_temp.keys())
    tasas_ordenadas = [datos_temp[t] for t in temps_ordenadas]
    labels_temps = [str(t) for t in temps_ordenadas]
    
    plt.figure(figsize=(8, 6))
    mejor_tasa = max(tasas_ordenadas)
    colores = ['#e74c3c' if tasa == mejor_tasa else '#3498db' for tasa in tasas_ordenadas]
    
    barras = plt.bar(labels_temps, tasas_ordenadas, color=colores, edgecolor='black', width=0.6)
    
    for rect in barras:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.title('Tasa de Éxito del Operador Paraphrase según Temperatura (GrIPS)', pad=20, fontsize=12)
    plt.xlabel('Temperatura del LLM ($T$)', fontsize=11)
    plt.ylabel('Tasa de Éxito (%)', fontsize=11)
    plt.ylim(0, max(tasas_ordenadas) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    ruta_out_dir = os.path.join(base_dir, 'graficos_finales')
    os.makedirs(ruta_out_dir, exist_ok=True)
    ruta_out = os.path.join(ruta_out_dir, 'exp4_tasa_exito_paraphrase.png')
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    print(f"Guardado: {ruta_out}")
    plt.close()

if __name__ == "__main__":
    graficar_tasa_exito_paraphrase()
