import time
from main import ejecutar_experimento
from scatter_search import ScatterSearch

def ejecutar_bateria_fase2():
    print("==================================================")
    print(" INICIANDO TEST A/B: COHERENCIA DE OPERADORES (3 EJECUCIONES)")
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
    for i in range(1, 4):
        sufijo = "" if i == 1 else f"_{i}"
        archivo_salida = f"resultado_coherencia_pura{sufijo}.json"
        
        print(f" -> Ejecutando corrida {i}/3: {archivo_salida}...")
        ejecutar_experimento(
            generaciones=10, 
            tamano_poblacion=40, 
            tamano_refset=10, 
            archivo_salida=archivo_salida
        )
    
    print("\n[PAUSA TÉCNICA] Dejando enfriar el equipo por 30 segundos...\n")
    time.sleep(30)
    
    # --- PRUEBA B: CRUCE COHERENTE (Con LLM Corrector) ---
    print(">>> LANZANDO PRUEBA B: CRUCE COHERENTE (Llama 3 arregla la gramática)")
    
    # Hackeamos la clase ScatterSearch para que use el cruce coherente
    ScatterSearch.combinar_soluciones = ScatterSearch.combinar_soluciones_coherente
    
    for i in range(1, 4):
        sufijo = "" if i == 1 else f"_{i}"
        archivo_salida = f"resultado_coherencia_llm{sufijo}.json"
        
        print(f" -> Ejecutando corrida {i}/3: {archivo_salida}...")
        ejecutar_experimento(
            generaciones=10, 
            tamano_poblacion=40, 
            tamano_refset=10, 
            archivo_salida=archivo_salida
        )
        
    tiempo_total = time.time() - tiempo_global_inicio
    print(f"\nBatería Fase 2 completada en {tiempo_total/60:.2f} minutos.")

if __name__ == "__main__":
    ejecutar_bateria_fase2()