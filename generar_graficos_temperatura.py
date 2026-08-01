import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generar_graficos():
    print("Generando gráficos de Temperatura...")
    
    # ---------------------------------------------------------
    # Gráfico 1: Experimento 5 (Inicialización) - Diversidad
    # ---------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    ruta_exp5 = os.path.join(base_dir, "exp5_inicializacion.json")
    if os.path.exists(ruta_exp5):
        with open(ruta_exp5, "r", encoding="utf-8") as f:
            d5 = json.load(f)
            
        temperaturas_5 = []
        diversidades = []
        
        for t_key in sorted(d5.keys()): # conf_T0.1, conf_T0.3...
            runs = d5[t_key]
            temp_val = float(t_key.replace("conf_T", ""))
            div_promedio = np.mean([r['diversidad_lexica'] for r in runs])
            
            temperaturas_5.append(str(temp_val))
            diversidades.append(div_promedio)
            
        plt.figure(figsize=(8, 6))
        mejor_div = max(diversidades)
        colores_5 = ['#e74c3c' if div == mejor_div else '#3498db' for div in diversidades]
        barras = plt.bar(temperaturas_5, diversidades, color=colores_5, edgecolor='black', width=0.6)
        
        # Añadir valores sobre las barras
        for rect in barras:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 2,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
                     
        plt.title('Experimento 5: Impacto de la Temperatura en la Diversidad Léxica (Fase 1)', pad=20, fontsize=12)
        plt.xlabel('Temperatura del LLM ($T$)', fontsize=11)
        plt.ylabel('Promedio de Palabras Únicas Generadas', fontsize=11)
        plt.ylim(0, max(diversidades) * 1.15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        ruta_out_dir = os.path.join(base_dir, 'graficos_finales')
        os.makedirs(ruta_out_dir, exist_ok=True)
        ruta_out_5 = os.path.join(ruta_out_dir, 'exp5_diversidad_lexica.png')
        plt.savefig(ruta_out_5, dpi=300, bbox_inches='tight')
        print(f"Guardado: {ruta_out_5}")
        plt.close()
        
    # ---------------------------------------------------------
    # Gráfico 2: Experimento 4 (GrIPS) - SBERT Final
    # ---------------------------------------------------------
    ruta_exp4 = os.path.join(base_dir, "exp4_temperaturas.json")
    ruta_exp1 = os.path.join(base_dir, "Resultados - Exp 1", "exp1_base.json")
    
    if os.path.exists(ruta_exp4) and os.path.exists(ruta_exp1):
        # Cargar Exp 4
        with open(ruta_exp4, "r", encoding="utf-8") as f:
            d4 = json.load(f)
            
        # Cargar Exp 1 (T=0.2 base)
        with open(ruta_exp1, "r", encoding="utf-8") as f:
            runs_base = json.load(f)
            # El SBERT final promedio del Exp 1 (Solo las primeras 3 corridas para ser justos con Exp 4)
            sbert_02 = np.mean([r['historial_convergencia'][-1]['mejor_sbert'] for r in runs_base[:3]])
            
        # Extraer SBERTs de Exp 4
        datos_temp = {}
        for t_key in d4.keys():
            temp_val = float(t_key.replace("conf_T", ""))
            runs = d4[t_key]
            sbert_prom = np.mean([r['historial_convergencia'][-1]['mejor_sbert'] for r in runs])
            datos_temp[temp_val] = sbert_prom
            
        # Insertar T=0.2
        datos_temp[0.2] = sbert_02
        
        # Ordenar para el gráfico
        temps_ordenadas = sorted(datos_temp.keys())
        sberts_ordenados = [datos_temp[t] for t in temps_ordenadas]
        labels_temps = [str(t) for t in temps_ordenadas]
        
        plt.figure(figsize=(8, 6))
        # Destacar el mejor
        mejor_sbert = max(sberts_ordenados)
        colores = ['#e74c3c' if sbert == mejor_sbert else '#3498db' for sbert in sberts_ordenados]
        
        barras2 = plt.bar(labels_temps, sberts_ordenados, color=colores, edgecolor='black', width=0.6)
        
        for rect in barras2:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
                     
        plt.title('Experimento 4: Calidad Semántica Final según Temperatura (Fase 2 - GrIPS)', pad=20, fontsize=12)
        plt.xlabel('Temperatura del LLM ($T$)', fontsize=11)
        plt.ylabel('Puntaje SBERT Final (Promedio)', fontsize=11)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        ruta_out_4 = os.path.join(ruta_out_dir, 'exp4_temperaturas_grips.png')
        plt.savefig(ruta_out_4, dpi=300, bbox_inches='tight')
        print(f"Guardado: {ruta_out_4}")
        plt.close()

if __name__ == "__main__":
    generar_graficos()
