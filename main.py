import json
import time
import copy
import random
from scatter_search import ScatterSearch
from grips import ModuloGrIPS

def ejecutar_experimento(
    generaciones=10, 
    tamano_poblacion=40, 
    tamano_refset=10, 
    archivo_salida="resultado.json", 
    validacion_cruzada=True,
    llm_model_name="llama3.1:8b",
    sbert_model_name="all-MiniLM-L6-v2",
    max_tokens_salida=150,
    semilla_global=None
):
    # ==========================================
    # ANCLAJE DE ALEATORIEDAD MAESTRA
    # ==========================================
    if semilla_global is not None:
        random.seed(semilla_global)
    else:
        random.seed()

    print(f"==================================================")
    print(f" INICIANDO EXPERIMENTO SCATTER SEARCH (GrIPS) ")
    print(f" Config: {generaciones} Gen | Pob: {tamano_poblacion} | RefSet: {tamano_refset}")
    print(f" Val Cruzada: {validacion_cruzada} | Semilla: {semilla_global}")
    print(f" Modelos: LLM={llm_model_name}, SBERT={sbert_model_name}")
    print(f"==================================================\n")

    tiempo_inicio = time.time()

    # 1. Inicializar Scatter Search con todos los parámetros controlados
    ss = ScatterSearch(
        tamano_poblacion=tamano_poblacion, 
        tamano_refset=tamano_refset,
        validacion_cruzada=validacion_cruzada,
        llm_model_name=llm_model_name,
        sbert_model_name=sbert_model_name,
        max_tokens_salida=max_tokens_salida,
        semilla_global=semilla_global
    )
    
    grips = ModuloGrIPS(ss.llm, ss.evaluador)

    # 2. Inicialización
    poblacion_inicial = ss.generar_poblacion_inicial()
    refset = ss.construir_refset(poblacion_inicial)
    poblacion_inicial_foto = copy.deepcopy(poblacion_inicial)
    
    mejor_global = refset[0]
    
    historial_convergencia = []
    historial_convergencia.append({
        "generacion": 0,
        "mejor_sbert": mejor_global.score_sbert,
        "promedio_sbert_refset": sum([s.score_sbert for s in refset]) / len(refset)
    })
    
    # 3. Bucle Principal de Generaciones
    for gen in range(1, generaciones + 1):
        print(f"\n--- GENERACIÓN {gen} ---")
        nuevos_hijos = []
        pares = ss.generar_pares(refset)
        print(f"  Combinando {len(pares)} pares del RefSet...")
        
        for p1, p2 in pares:
            hijos = ss.combinar_soluciones(p1, p2)
            for h in hijos:
                h_mejorado = grips.ejecutar_greedy(h, ss.textos_referencia)
                nuevos_hijos.append(h_mejorado)
            
        print(f"  Se generaron y procesaron por GrIPS {len(nuevos_hijos)} hijos nuevos.")
        
        poblacion_combinada = refset + nuevos_hijos
        refset = ss.construir_refset(poblacion_combinada)
        
        mejor_actual = refset[0]
        if mejor_actual.score_sbert > mejor_global.score_sbert:
            mejor_global = mejor_actual
            
        promedio_gen = sum([s.score_sbert for s in refset]) / len(refset)
        historial_convergencia.append({
            "generacion": gen,
            "mejor_sbert": mejor_actual.score_sbert,
            "promedio_sbert_refset": promedio_gen
        })
        print(f"  [Gen {gen} Fin] Mejor SBERT: {mejor_actual.score_sbert:.4f} | Promedio RefSet: {promedio_gen:.4f}")

    tiempo_fin = time.time()
    tiempo_total_minutos = round((tiempo_fin - tiempo_inicio) / 60.0, 2)
    print(f"\n================ EXPERIMENTO FINALIZADO ================")
    print(f"Tiempo total: {tiempo_total_minutos} minutos.")
    print(f"Mejor SBERT Global: {mejor_global.score_sbert:.4f}")
    
    # 5. Función auxiliar para serializar
    def serializar_poblacion(poblacion_lista):
        return [{
            "rol": sol.rol,
            "tarea": sol.tarea,
            "prompt": sol.prompt_completo,
            "tweet_generado": getattr(sol, 'dato_generado', ''),
            "tweet_real_match": getattr(sol, 'texto_referencia_match', ''),
            "sbert": getattr(sol, 'score_sbert', 0.0),
            "origen": sol.origen
        } for sol in poblacion_lista]

    # 6. Guardar Resultados Finales
    salida = {
        "configuracion": {
            "generaciones": generaciones,
            "tamano_poblacion": tamano_poblacion,
            "tamano_refset": tamano_refset,
            "validacion_cruzada": validacion_cruzada,
            "llm_model": llm_model_name,
            "sbert_model": sbert_model_name,
            "semilla_global": semilla_global,
            "tiempo_ejecucion_minutos": tiempo_total_minutos
        },
        "historial_convergencia": historial_convergencia,
        "poblacion_inicial": serializar_poblacion(poblacion_inicial_foto),
        "mejor_individuo": { 
            "rol": mejor_global.rol,
            "tarea": mejor_global.tarea,
            "prompt": mejor_global.prompt_completo,
            "tweet_generado": mejor_global.dato_generado,
            "tweet_real_match": getattr(mejor_global, 'texto_referencia_match', ''),
            "sbert": mejor_global.score_sbert,
            "origen": mejor_global.origen
        },
        "refset_final": serializar_poblacion(refset),
        "stats_grips": grips.stats_operadores
    }
    
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=4, ensure_ascii=False)
        
    print(f"Resultados guardados en {archivo_salida}")
    return salida

if __name__ == "__main__":
    ejecutar_experimento(generaciones=1, tamano_poblacion=4, tamano_refset=2, archivo_salida="test_run.json", semilla_global=42)