import json
import os

def calcular_jaccard(str1: str, str2: str) -> float:
    """
    Calcula la similitud de Jaccard entre dos textos basándose en la superposición de palabras.
    """
    if not str1 or not str2:
        return 0.0
        
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if not set1 or not set2:
        return 0.0
        
    interseccion = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(interseccion) / len(union)

def analizar_jaccard_archivo(ruta_json: str):
    """
    Lee un archivo de resultados JSON del experimento y calcula estadísticas de Jaccard.
    """
    if not os.path.exists(ruta_json):
        print(f"Archivo no encontrado: {ruta_json}")
        return None
        
    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    resultados = {
        "archivo": ruta_json,
        "configuracion_original": data.get("configuracion", {}),
        "mejor_individuo": {},
        "poblacion_inicial": {},
        "refset_final": {}
    }
    
    # 1. Analizar Mejor Individuo
    mejor_ind = data.get("mejor_individuo", {})
    if mejor_ind:
        tweet_gen = mejor_ind.get("tweet_generado", "")
        tweet_real = mejor_ind.get("tweet_real_match", "")
        jaccard = calcular_jaccard(tweet_gen, tweet_real)
        
        resultados["mejor_individuo"] = {
            "sbert": mejor_ind.get("sbert", 0),
            "jaccard": jaccard
        }
        
    # 2. Analizar Población Inicial
    poblacion = data.get("poblacion_inicial", [])
    if poblacion:
        jaccards_pob = []
        for ind in poblacion:
            tweet_gen = ind.get("tweet_generado", "")
            tweet_real = ind.get("tweet_real_match", "")
            jaccards_pob.append(calcular_jaccard(tweet_gen, tweet_real))
            
        resultados["poblacion_inicial"] = {
            "jaccard_promedio": sum(jaccards_pob) / len(jaccards_pob) if jaccards_pob else 0,
            "jaccard_max": max(jaccards_pob) if jaccards_pob else 0
        }
        
    # 3. Analizar Refset Final
    refset = data.get("refset_final", [])
    if refset:
        jaccards_ref = []
        for ind in refset:
            tweet_gen = ind.get("tweet_generado", "")
            tweet_real = ind.get("tweet_real_match", "")
            jaccards_ref.append(calcular_jaccard(tweet_gen, tweet_real))
            
        resultados["refset_final"] = {
            "jaccard_promedio": sum(jaccards_ref) / len(jaccards_ref) if jaccards_ref else 0,
            "jaccard_max": max(jaccards_ref) if jaccards_ref else 0
        }
        
    return resultados

def ejecutar_analisis(archivos_a_analizar, archivo_salida="resultados_jaccard.json"):
    print(f"--- INICIANDO ANÁLISIS JACCARD POST-CORRIDA ---")
    resultados_totales = []
    
    for archivo in archivos_a_analizar:
        print(f"Procesando: {archivo}...")
        resultado = analizar_jaccard_archivo(archivo)
        if resultado:
            resultados_totales.append(resultado)
            
    # Guardar resultados
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)
        
    print(f"\nAnálisis completado. Resultados guardados en: {archivo_salida}")

if __name__ == "__main__":
    # Define aquí los archivos generados por tu experimento principal que quieres analizar
    archivos_json = [
        "resultado_eval_estandar.json", 
        "resultado_eval_cruzada.json",
        "test_run.json"
    ]
    
    ejecutar_analisis(archivos_json)
