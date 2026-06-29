import json
import os
import matplotlib.pyplot as plt

def generar_torta_grips():
    ruta_json = os.path.join(os.path.dirname(__file__), '..', 'exp1_base.json')
    ruta_out = os.path.join(os.path.dirname(__file__), '..', 'torta_grips.png')
    
    with open(ruta_json, "r", encoding="utf-8") as f:
        runs = json.load(f)
        
    total_stats = {"grips_delete": 0, "grips_swap": 0, "grips_paraphrase": 0}
    
    # Sumar éxitos de las 5 corridas
    for run in runs:
        if "stats_grips" in run:
            for op, stats in run["stats_grips"].items():
                if op in total_stats:
                    total_stats[op] += stats.get("exitos", 0)
                
    labels = list(total_stats.keys())
    sizes = list(total_stats.values())
    
    if sum(sizes) == 0:
        print("No hay éxitos de GrIPS registrados.")
        return
        
    plt.figure(figsize=(8, 8))
    colores = ['#ff9999','#66b3ff','#99ff99']
    
    # Mostrar valores y porcentajes
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute:d})"
        
    import numpy as np
    plt.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes), 
            startangle=140, colors=colores, wedgeprops={'edgecolor': 'black'})
            
    plt.title('Rendimiento Aislado: Mejoras Netas al Fitness por Operador GrIPS')
    plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {ruta_out}")

if __name__ == "__main__":
    generar_torta_grips()
