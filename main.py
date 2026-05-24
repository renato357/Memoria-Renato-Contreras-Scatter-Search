import json
import time
from scatter_search import ScatterSearch
from grips import ModuloGrIPS
from dataset import DatasetManager

def ejecutar_experimento(generaciones=3, tamano_poblacion=10, tamano_refset=4, archivo_salida="resultado_experimento.json"):
    print("==================================================")
    print(f" INICIANDO EXPERIMENTO SS-GrIPS (CON TELEMETRÍA)")
    print(f" Generaciones: {generaciones} | Población: {tamano_poblacion} | RefSet: {tamano_refset}")
    print("==================================================\n")
    
    tiempo_inicio = time.time()
    
    # 1. Instanciamos los módulos
    ss = ScatterSearch(tamano_poblacion=tamano_poblacion, tamano_refset=tamano_refset)
    grips = ModuloGrIPS(ss.llm, ss.evaluador)
    
    # Inicializamos la lista del Historiador
    historial_convergencia = []
    
    # 2. Inicialización
    poblacion_inicial = ss.generar_poblacion_inicial()
    refset = ss.construir_refset(poblacion_inicial)
    
    mejor_global = refset[0] # El top 1 actual
    
    # 3. Bucle Evolutivo
    for g in range(generaciones):
        print(f"\n================ GENERACIÓN {g+1}/{generaciones} ================")
        
        # Generar pares y cruzar
        pares = ss.generar_pares(refset)
        hijos_nueva_gen = []
        
        print(f"[Fase 3] Cruzando {len(pares)} pares de padres...")
        for p1, p2 in pares:
            hijos = ss.combinar_soluciones(p1, p2)
            hijos_nueva_gen.extend(hijos)
            
        print(f"[Fase 4] Aplicando GrIPS a {len(hijos_nueva_gen)} hijos nuevos...")
        hijos_mejorados = []
        for i, hijo in enumerate(hijos_nueva_gen):
            print(f"\n  -- Optimizando Hijo {i+1}/{len(hijos_nueva_gen)} [{hijo.origen}] --")
            hijo_optimizado = grips.ejecutar_greedy(hijo, ss.textos_referencia)
            hijos_mejorados.append(hijo_optimizado)
            
        # 4. Actualización del RefSet
        print("\n[Fase 5] Actualizando el Conjunto de Referencia (RefSet)...")
        pool_total = refset + hijos_mejorados
        
        # Evitamos re-evaluar a los que ya tienen SBERT calculado
        for sol in pool_total:
            if sol.score_sbert == 0.0:
                ss.evaluar_solucion(sol)
                
        # Lógica de actualización (Mitad SBERT, Mitad BLEU)
        pool_total.sort(key=lambda x: x.score_sbert, reverse=True)
        nuevo_elite = pool_total[:ss.b_elite]
        restantes = pool_total[ss.b_elite:]
        
        # --- CAMBIO CLAVE: BLEU SOBRE LA DATA GENERADA (TWEETS) ---
        textos_tweets_elite = [sol.dato_generado for sol in nuevo_elite]
        for sol in restantes:
            sol.score_bleu = ss.evaluador.calcular_diversidad_bleu(sol.dato_generado, textos_tweets_elite)
            
        restantes.sort(key=lambda x: x.score_bleu)
        nuevo_div = restantes[:ss.b_div]
        
        refset = nuevo_elite + nuevo_div
        
        # Tracking del mejor global
        if refset[0].score_sbert > mejor_global.score_sbert:
            mejor_global = refset[0]
            print(f"\n>>> ¡NUEVO RÉCORD GLOBAL! SBERT: {mejor_global.score_sbert:.4f} <<<")
            
        print(f"Mejor SBERT de la Gen {g+1}: {refset[0].score_sbert:.4f}")

        # --- TELEMETRÍA DEL HISTORIADOR POR GENERACIÓN ---
        max_sbert = refset[0].score_sbert
        avg_sbert_elite = sum(sol.score_sbert for sol in nuevo_elite) / len(nuevo_elite)
        avg_bleu_diversidad = sum(sol.score_bleu for sol in nuevo_div) / len(nuevo_div)
        avg_len_tarea = sum(len(sol.tarea.split()) for sol in refset) / len(refset)
        
        historial_convergencia.append({
            "generacion": g + 1,
            "max_sbert": max_sbert,
            "avg_sbert_elite": avg_sbert_elite,
            "avg_bleu_diversidad": avg_bleu_diversidad,
            "avg_len_tarea": avg_len_tarea
        })

    tiempo_total = (time.time() - tiempo_inicio) / 60
    print(f"\n================ EXPERIMENTO FINALIZADO ================")
    print(f"Tiempo total: {tiempo_total:.2f} minutos.")
    
    # 5. Guardar Resultados incluyendo el historial de convergencia
    salida = {
        "configuracion": {
            "generaciones": generaciones,
            "tamano_poblacion": tamano_poblacion,
            "tamano_refset": tamano_refset
        },
        "historial_convergencia": historial_convergencia,
        "mejor_individuo": {
            "rol": mejor_global.rol,
            "tarea": mejor_global.tarea,
            "prompt": mejor_global.prompt_completo,
            "tweet_generado": mejor_global.dato_generado,
            "sbert": mejor_global.score_sbert,
            "bleu": mejor_global.score_bleu,
            "origen": mejor_global.origen
        },
        "refset_final": [
            {
                "tarea": sol.tarea, 
                "sbert": sol.score_sbert, 
                "bleu": sol.score_bleu, 
                "origen": sol.origen
            } for sol in refset
        ]
    }
    
    with open(archivo_salida, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=4, ensure_ascii=False)
        
    print(f"Resultados guardados en {archivo_salida}")

if __name__ == "__main__":
    print("INICIANDO BATERÍA DE EXPERIMENTACIÓN OPTIMIZADA SS-GrIPS")
    print("Ejecutando configuración óptima para recolección de curvas.\n")
    
    # Configurado únicamente con la versión Media (Ganadora del Análisis de Sensibilidad)
    experimentos = [
        {
            "nombre": "resultado_media.json",
            "gen": 10,
            "pop": 40,
            "ref": 10
        }
    ]
    
    tiempo_global_inicio = time.time()
    
    for exp in experimentos:
        try:
            ejecutar_experimento(
                generaciones=exp["gen"], 
                tamano_poblacion=exp["pop"], 
                tamano_refset=exp["ref"], 
                archivo_salida=exp["nombre"]
            )
            print(f"\nExperimento {exp['nombre']} finalizado con éxito.\n")
            time.sleep(10) 
            
        except Exception as e:
            print(f"\nError en {exp['nombre']}: {e}")
            
    tiempo_global_fin = (time.time() - tiempo_global_inicio) / 3600
    print("BATERÍA COMPLETA")
    print(f"Tiempo total de ejecución: {tiempo_global_fin:.2f} horas.")