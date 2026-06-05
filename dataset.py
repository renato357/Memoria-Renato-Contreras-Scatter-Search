import pandas as pd
import random
import os

class DatasetManager:
    def __init__(self, ruta_csv="./corpus (1).csv"):
        """
        Administrador del dataset adaptado para control centralizado de parámetros.
        La ruta por defecto apunta al archivo real verificado.
        """
        self.ruta_csv = ruta_csv

    def obtener_muestra_referencia(self, n=50, excluir_textos=None) -> list:
        """
        Carga una muestra aleatoria de 'n' textos desde el archivo CSV.
        
        Parámetros:
        - n: Cantidad de textos a extraer.
        - excluir_textos: Lista de strings que NO deben seleccionarse (evita solapamiento 
                          en validación cruzada excluyente).
        """
        print(f"Cargando muestra de {n} textos de referencia desde el dataset...")
        try:
            df = pd.read_csv(self.ruta_csv, header=None, encoding='utf-8', engine='python')
            textos_reales = df.iloc[:, 0].dropna().astype(str).tolist()
            
            # Filtrar exclusiones para mantener sets totalmente independientes
            if excluir_textos is not None:
                textos_reales = [t for t in textos_reales if t not in excluir_textos]
            
            # Como la semilla global ya se fijó en main.py, esto avanzará la secuencia
            # sin repetir los mismos textos de la extracción anterior.
            muestra = random.sample(textos_reales, min(n, len(textos_reales)))
            muestra = [texto.strip() for texto in muestra if texto.strip()]
            
            return muestra

        except FileNotFoundError:
            print(f"[Error] No se encontró el archivo de datos en la ruta: '{self.ruta_csv}'")
            return []
        except Exception as e:
            print(f"[Error] Fallo al procesar el dataset: {e}")
            return []

if __name__ == "__main__":
    # Prueba local de sanidad y funcionamiento aislado
    dm = DatasetManager()
    test_1 = dm.obtener_muestra_referencia(n=3)
    print("Muestra extraída:", test_1)