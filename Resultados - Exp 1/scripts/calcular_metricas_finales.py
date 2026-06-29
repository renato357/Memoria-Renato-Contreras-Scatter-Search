import json
import os
import numpy as np

def calcular_metricas():
    ruta_json = os.path.join(os.path.dirname(__file__), '..', 'exp1_base.json')
    
    with open(ruta_json, "r", encoding="utf-8") as f:
        runs = json.load(f)
        
    sberts = []
    jaccards = []
    tiempos = []
    
    for run in runs:
        # Tiempo
        tiempo = run['configuracion']['tiempo_ejecucion_minutos']
        tiempos.append(tiempo)
        
        # SBERT (última generación)
        sbert_run = run['historial_convergencia'][-1]['mejor_sbert']
        sberts.append(sbert_run)
        
        # Jaccard del RefSet final
        run_jaccards = []
        for ind in run.get('refset_final', []):
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
            
    print("=== MÉTRICAS FINALES (EXPERIMENTO 1: BASELINE) ===")
    print(f"Número de Corridas (N): {len(runs)}")
    print(f"Promedio SBERT Final:  {np.mean(sberts):.4f}  (std: {np.std(sberts):.4f})")
    print(f"Promedio Jaccard:      {np.mean(jaccards):.4f}  (std: {np.std(jaccards):.4f})")
    print(f"Tiempo Promedio:       {np.mean(tiempos):.2f} minutos")

if __name__ == "__main__":
    calcular_metricas()
