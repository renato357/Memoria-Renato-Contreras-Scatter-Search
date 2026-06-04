import pandas as pd
import random
import os

class DatasetManager:
    def __init__(self, ruta_csv='Mental-Health-Twitter.csv'):
        self.ruta_csv = ruta_csv

    def obtener_muestra_referencia(self, n=50, semilla=None, excluir_textos=None) -> list:
        """
        Carga una muestra aleatoria de 'n' tweets desde el archivo CSV.
        Si se pasa 'semilla', la fija. Si es None, extrae una muestra distinta en cada ejecución.
        'excluir_textos' permite pasar una lista de textos que NO deben ser seleccionados,
        útil para separar el set de contexto del LLM y el set de evaluación SBERT.
        """
        print(f"Cargando muestra de {n} textos de referencia desde el dataset...")
        try:
            df = pd.read_csv(self.ruta_csv, header=None, encoding='utf-8', engine='python')
            textos_reales = df.iloc[:, 0].dropna().astype(str).tolist()
            
            # Filtramos los textos que queremos excluir (para que sean muestras 100% distintas)
            if excluir_textos is not None:
                textos_reales = [t for t in textos_reales if t not in excluir_textos]
            
            # Lógica de aleatoriedad cruzada
            if semilla is not None:
                random.seed(semilla)
            else:
                random.seed() # Libera la semilla para que sea verdaderamente aleatorio
            
            muestra = random.sample(textos_reales, min(n, len(textos_reales)))
            muestra = [texto.strip() for texto in muestra if texto.strip()]
            
            return muestra

        except FileNotFoundError:
            print(f"[Error] No se encontró el archivo '{self.ruta_csv}'. Asegúrate de que esté en la carpeta.")
            return []
        except Exception as e:
            print(f"[Error] Fallo al procesar el dataset: {e}")
            return []

if __name__ == "__main__":
    # Prueba rápida
    dm = DatasetManager()
    muestra_1 = dm.obtener_muestra_referencia(n=5)
    print("Muestra 1 (Ejemplos):", muestra_1)
    
    muestra_2 = dm.obtener_muestra_referencia(n=5, excluir_textos=muestra_1)
    print("Muestra 2 (Evaluación, distintos a la 1):", muestra_2)