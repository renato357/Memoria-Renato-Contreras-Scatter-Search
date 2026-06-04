import json
import time
import copy
from scatter_search import ScatterSearch
from grips import ModuloGrIPS

def ejecutar_experimento(generaciones=10, tamano_poblacion=40, tamano_refset=10, archivo_salida="resultado.json", validacion_cruzada=True):
    print(f"==================================================")
    print(f" INICIANDO EXPERIMENTO SCATTER SEARCH (GrIPS) ")
    print(f" Config: {generaciones} Gen | Pob: {tamano_poblacion} | RefSet: {tamano_refset} | Val Cruzada: {validacion_cruzada}")
    print(f"==================================================\n")

    tiempo_inicio = time.time()

    # 1. Inicializar Scatter Search (ahora recibe la orden de validación)
    ss = ScatterSearch(
        tamano_poblacion=tamano_poblacion, 
        tamano_refset=tamano_refset,
        validacion_cruzada=validacion_cruzada
    )
    grips = ModuloGrIPS()

    # 2. Inicialización
    # (Ojo: run_fase1.py sobreescribe esta función en tiempo de ejecución si se usa Llama 3)
    poblacion_inicial = ss.generar_poblacion_inicial()
    
    # Evaluamos y construimos el RefSet inicial. 
    # Al pasar por aquí, a los 40 individuos se les calcula el SBERT.
    refset = ss.construir_refset(poblacion_inicial)
    
    # [NUEVO] Guardamos la foto completa de los 40 individuos iniciales (ya evaluados).
    # Usamos deepcopy porque los objetos mutarán y se cruzarán en las siguientes generaciones.
    poblacion_inicial_foto = copy.deepcopy(poblacion_inicial)
    
    mejor_global = refset[0] # El top 1 actual
    
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
                # Mutación GrIPS con probabilidad interna
                if grips.aplicar_mutacion():
                    h.tarea = grips.mutar(h.tarea)
                    h.origen += "_mutated"
            nuevos_hijos.extend(hijos)
            
        print(f"  Se generaron {len(nuevos_hijos)} hijos nuevos.")
        
        # Unimos RefSet actual con los nuevos hijos y reconstruimos la élite
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

    # 4. Fin del Tiempo
    tiempo_fin = time.time()
    tiempo_total_minutos = round((tiempo_fin - tiempo_inicio) / 60.0, 2)
    print(f"\n================ EXPERIMENTO FINALIZADO ================")
    print(f"Tiempo total: {tiempo_total_minutos} minutos.")
    print(f"Mejor SBERT Global: {mejor_global.score_sbert:.4f}")
    
    # 5. Función auxiliar para convertir las listas de objetos en diccionarios para el JSON
    def serializar_poblacion(poblacion_lista):
        return [{
            "rol": sol.rol,
            "tarea": sol.tarea,
            "prompt": sol.prompt_completo,
            "tweet_generado": getattr(sol, 'dato_generado', ''),
            "sbert": getattr(sol, 'score_sbert', 0.0),
            "bleu": getattr(sol, 'score_bleu', 0.0),
            "origen": sol.origen
        } for sol in poblacion_lista]

    # 6. Guardar Resultados Finales
    salida = {
        "configuracion": {
            "generaciones": generaciones,
            "tamano_poblacion": tamano_poblacion,
            "tamano_refset": tamano_refset,
            "validacion_cruzada": validacion_cruzada, # Registramos qué método se usó
            "tiempo_ejecucion_minutos": tiempo_total_minutos
        },
        "historial_convergencia": historial_convergencia,
        "poblacion_inicial": serializar_poblacion(poblacion_inicial_foto), # Los 40 iniciales
        "mejor_individuo": { # Mantenido por compatibilidad con graficador
            "rol": mejor_global.rol,
            "tarea": mejor_global.tarea,
            "prompt": mejor_global.prompt_completo,
            "tweet_generado": mejor_global.dato_generado,
            "sbert": mejor_global.score_sbert,
            "bleu": getattr(mejor_global, 'score_bleu', 0.0),
            "origen": mejor_global.origen
        },
        "refset_final": serializar_poblacion(refset) # Los 10 finales
    }
    
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=4, ensure_ascii=False)
        
    print(f"Resultados guardados en {archivo_salida}")

if __name__ == "__main__":
    # Corrida de prueba rápida por si ejecutas main.py directo
    ejecutar_experimento(generaciones=3, tamano_poblacion=10, tamano_refset=4, archivo_salida="test_run.json", validacion_cruzada=True)