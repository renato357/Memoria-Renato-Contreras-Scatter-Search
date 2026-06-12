import json
import matplotlib.pyplot as plt

def cargar_datos(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    generaciones = []
    mejores_sbert = []
    for entrada in datos['historial_convergencia']:
        generaciones.append(entrada['generacion'])
        mejores_sbert.append(entrada['mejor_sbert'])
    return generaciones, mejores_sbert

ruta_estandar = 'resultado_eval_estandar.json'
ruta_cruzada = 'resultado_eval_cruzada.json'

gen_est, sbert_est = cargar_datos(ruta_estandar)
gen_cruz, sbert_cruz = cargar_datos(ruta_cruzada)

plt.figure(figsize=(10, 6))

plt.plot(gen_est, sbert_est, marker='o', label='Estrategia Estándar', linestyle='-', color='blue')
plt.plot(gen_cruz, sbert_cruz, marker='s', label='Estrategia Cruzada', linestyle='--', color='red')

plt.title('Evolución del Mejor SBERT por Generación')
plt.xlabel('Generación')
plt.ylabel('Mejor SBERT')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('evolucion_mejor_sbert.png')
print("Gráfico guardado como 'evolucion_mejor_sbert.png'")
