import time
from main import ejecutar_experimento
from scatter_search import ScatterSearch

def ejecutar_bateria_fase2():
    print("==================================================")
    print(" INICIANDO TEST A/B: COHERENCIA DE OPERADORES")
    print("==================================================\n")
    
    tiempo_global_inicio = time.time()
    
    # ---------------------------------------------------------
    # REGLA DE ORO: Usamos el ganador de la Fase 1 para ambas pruebas
    # ---------------------------------------------------------
    ScatterSearch.generar_poblacion_inicial = ScatterSearch.generar_poblacion_inicial_llm
    
    # Guardamos la función original de cruce por si acaso
    metodo_cruce_puro = ScatterSearch.combinar_soluciones
    
    # --- PRUEBA A: CRUCE PURO (Sin LLM Corrector) ---
    print(">>> LANZANDO PRUEBA A: CRUCE PURO (Rápido, pero gramática rota)")
    ejecutar_experimento(
        generaciones=10, 
        tamano_poblacion=40, 
        tamano_refset=10, 
        archivo_salida="resultado_coherencia_pura.json"
    )
    
    print("\n[PAUSA TÉCNICA] Dejando enfriar el equipo por 30 segundos...\n")
    time.sleep(30)
    
    # --- PRUEBA B: CRUCE COHERENTE (Con LLM Corrector) ---
    print(">>> LANZANDO PRUEBA B: CRUCE COHERENTE (Llama 3 arregla la gramática)")
    
    # Hackeamos la clase ScatterSearch para que use el cruce coherente de tu profe
    ScatterSearch.combinar_soluciones = ScatterSearch.combinar_soluciones_coherente
    
    ejecutar_experimento(
        generaciones=10, 
        tamano_poblacion=40, 
        tamano_refset=10, 
        archivo_salida="resultado_coherencia_llm.json"
    )
    
    # Restauramos el método original al terminar
    ScatterSearch.combinar_soluciones = metodo_cruce_puro
    
    tiempo_global_fin = (time.time() - tiempo_global_inicio) / 3600
    print("\n==================================================")
    print(f" BATERÍA FASE 2 COMPLETADA EN {tiempo_global_fin:.2f} HORAS.")
    print("==================================================")

if __name__ == "__main__":
    ejecutar_bateria_fase2()