import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generar_comparativa_leakage():
    ruta_exp1 = os.path.join(os.path.dirname(__file__), '..', '..', 'Resultados - Exp 1', 'exp1_base.json')
    ruta_exp3 = os.path.join(os.path.dirname(__file__), '..', 'exp3_cruzada_off.json')
    ruta_out = os.path.join(os.path.dirname(__file__), '..', 'comparativa_leakage.png')
    
    def get_metrics(archivo, limit=3):
        with open(archivo, "r", encoding="utf-8") as f:
            runs = json.load(f)
            
        # Tomamos solo el límite de corridas indicado para ser justos (N=3 vs N=3)
        runs_limitadas = runs[:limit]
        
        sberts = [r['historial_convergencia'][-1]['mejor_sbert'] for r in runs_limitadas]
        
        jaccards = []
        for run in runs_limitadas:
            refset = run.get('refset_final', [])
            run_jaccards = []
            for ind in refset:
                gen = ind.get('tweet_generado', '')
                real = ind.get('tweet_real_match', '')
                if gen and real:
                    set1 = set(gen.lower().split())
                    set2 = set(real.lower().split())
                    union_len = len(set1.union(set2))
                    j = len(set1.intersection(set2)) / union_len if union_len else 0
                    run_jaccards.append(j)
            if run_jaccards:
                jaccards.append(np.mean(run_jaccards))
                
        return np.mean(sberts), np.mean(jaccards) if jaccards else 0

    sbert_on, jaccard_on = get_metrics(ruta_exp1, limit=3)
    sbert_off, jaccard_off = get_metrics(ruta_exp3, limit=3)
    
    labels = ['Validación Cruzada (ON)', 'Data Leakage (OFF)']
    sberts = [sbert_on, sbert_off]
    jaccards = [jaccard_on, jaccard_off]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, sberts, width, label='SBERT Score', color='#3498db', edgecolor='black')
    rects2 = ax.bar(x + width/2, jaccards, width, label='Jaccard Index (Plagio)', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Scores Promedio (N=3)')
    ax.set_title('Impacto del Data Leakage en Métricas Finales\n(Promedio de las primeras 3 corridas)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.legend(loc='upper center')
    ax.set_ylim(0, 1.0)
    
    # Añadir valores sobre las barras
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {ruta_out}")
    print(f"\nResultados (N=3):")
    print(f"ON  -> SBERT: {sbert_on:.4f}, Jaccard: {jaccard_on:.4f}")
    print(f"OFF -> SBERT: {sbert_off:.4f}, Jaccard: {jaccard_off:.4f}")

if __name__ == "__main__":
    generar_comparativa_leakage()
