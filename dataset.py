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
        print(f"Cargando muestra de {n} textos de referencia desde el dataset...")
        try:
            df = pd.read_csv(self.ruta_csv, header=None, encoding='utf-8', engine='python')
            
            # PARCHE: Limpiamos los espacios primero para que la exclusión funcione
            textos_reales = df.iloc[:, 0].dropna().astype(str).tolist()
            textos_reales = [t.strip() for t in textos_reales if t.strip()]
            
            # Ahora sí la exclusión será matemáticamente estricta
            if excluir_textos is not None:
                textos_reales = [t for t in textos_reales if t not in excluir_textos]
            
            muestra = random.sample(textos_reales, min(n, len(textos_reales)))
            return muestra

        except Exception as e:
            print(f"[Error] Fallo al procesar el dataset: {e}")
            return []

if __name__ == "__main__":
    # Prueba local de sanidad y funcionamiento aislado
    dm = DatasetManager()
    test_1 = dm.obtener_muestra_referencia(n=3)
    print("Muestra extraída:", test_1)