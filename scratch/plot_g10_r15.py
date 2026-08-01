import json
import matplotlib.pyplot as plt
import os
import numpy as np

def graficar_convergencia_g10_r15():
    ruta = os.path.join("Resultados - Exp 2", "exp2_sensibilidad.json")
    
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Obtener datos de conf_G10_R15, que es una lista de ejecuciones (corridas)
            conf_list = data.get("conf_G10_R15")
            if not conf_list or not isinstance(conf_list, list):
                print("No se encontró conf_G10_R15 como lista en el archivo.")
                return
                
            print(f"Calculando promedio de {len(conf_list)} corridas...")
            
            num_generaciones = len(conf_list[0].get("historial_convergencia", []))
            mejor_sbert_all = np.zeros(num_generaciones)
            
            # Sumamos los valores de todas las corridas
            for conf_data in conf_list:
                historial = conf_data.get("historial_convergencia", [])
                for i, x in enumerate(historial):
                    mejor_sbert_all[i] += x.get('mejor_sbert', x.get('max_sbert', 0))
                    
            # Dividimos por el número de corridas para obtener el promedio
            mejor_sbert_promedio = mejor_sbert_all / len(conf_list)
            generaciones = list(range(num_generaciones))
            
            plt.figure(figsize=(10, 6))
            
            plt.plot(generaciones, mejor_sbert_promedio, 'b-o', label='Mejor SBERT (Promedio)', linewidth=2, markersize=6)
            
            plt.title('Convergencia de SBERT (Exp 2 - G10 R15) - Promedio de 3 Corridas', fontsize=14, fontweight='bold')
            plt.xlabel('Generación', fontsize=12)
            plt.ylabel('SBERT Máximo (Calidad Semántica)', fontsize=12)
            
            plt.xticks(generaciones)
            
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            nombre_salida = os.path.join("Resultados - Exp 2", "convergencia_G10_R15_promedio.png")
            plt.savefig(nombre_salida, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado exitosamente como: {nombre_salida}")
            
    except Exception as e:
        print(f"Error procesando los datos: {e}")

if __name__ == "__main__":
    graficar_convergencia_g10_r15()
