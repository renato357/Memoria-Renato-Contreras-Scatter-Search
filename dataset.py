import pandas as pd
import random

class DatasetManager:
    def __init__(self, ruta_csv="./corpus (1).csv"):
        """
        Gestor del dataset real. 
        """
        self.ruta_csv = ruta_csv

    def obtener_muestra_referencia(self, n=50, semilla=42) -> list:
        """
        Carga una muestra aleatoria de 'n' tweets desde el archivo CSV.
        Fija una semilla para garantizar que el análisis de sensibilidad sea reproducible.
        """
        print(f"Cargando muestra de {n} textos de referencia desde el dataset...")
        try:
            # Leemos el CSV. Como mencionas que es texto directo, asumimos header=None
            # y tomamos la primera columna.
            df = pd.read_csv(self.ruta_csv, header=None, encoding='utf-8', engine='python')
            
            # Extraemos la primera columna, botamos nulos y pasamos a lista
            textos_reales = df.iloc[:, 0].dropna().astype(str).tolist()
            
            # Fijamos la semilla de aleatoriedad
            random.seed(semilla)
            
            # Tomamos la muestra
            muestra = random.sample(textos_reales, min(n, len(textos_reales)))
            
            # Limpiamos espacios extra al inicio y final por si acaso
            muestra = [texto.strip() for texto in muestra if texto.strip()]
            
            return muestra

        except FileNotFoundError:
            print(f"[Error] No se encontró el archivo '{self.ruta_csv}'. Asegúrate de que esté en la carpeta.")
            return []
        except Exception as e:
            print(f"[Error] Fallo al procesar el dataset: {e}")
            return []

# --- PRUEBA RÁPIDA ---
if __name__ == "__main__":
    gestor = DatasetManager("corpus (1).csv")
    # Pedimos solo 5 para probar que funciona rápido
    textos_prueba = gestor.obtener_muestra_referencia(n=5)
    
    print("\nTextos de muestra obtenidos:")
    for i, txt in enumerate(textos_prueba):
        print(f"{i+1}. {txt}")