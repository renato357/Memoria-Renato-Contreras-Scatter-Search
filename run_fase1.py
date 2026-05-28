import time
from main import ejecutar_experimento
from scatter_search import ScatterSearch

# Un pequeño "hack" (Monkey Patch) para cambiar temporalmente la inicialización de la clase SS
# sin ensuciar tu código original con variables IF complejas.

def ejecutar_bateria_fase1():
    print("==================================================")
    print(" INICIANDO TEST A/B: POBLACIÓN INICIAL (5 EJECUCIONES)")
    print("==================================================\n")
    
    tiempo_global_inicio = time.time()
    
    # --- PRUEBA A: EL BASELINE (10x10) ---
    print(">>> LANZANDO PRUEBA A: INICIALIZACIÓN DETERMINISTA (10x10)")
    for i in range(1, 6):
        # Si es la primera corrida, no lleva número. Si no, lleva _2, _3, etc.
        sufijo = "" if i == 1 else f"_{i}"
        archivo_salida = f"resultado_init_10x10{sufijo}.json"
        
        print(f" -> Ejecutando corrida {i}/5: {archivo_salida}...")
        ejecutar_experimento(
            generaciones=10, 
            tamano_poblacion=40, 
            tamano_refset=10, 
            archivo_salida=archivo_salida
        )
    
    print("\n[PAUSA TÉCNICA] Dejando enfriar el equipo por 30 segundos...\n")
    time.sleep(30)
    
    # --- PRUEBA B: EL LLM (Meta-Prompting) ---
    print(">>> LANZANDO PRUEBA B: INICIALIZACIÓN ESTOCÁSTICA (Llama 3)")
    
    # Hackeamos la clase ScatterSearch en tiempo de ejecución para que use la variante
    ScatterSearch.generar_poblacion_inicial = ScatterSearch.generar_poblacion_inicial_llm
    
    for i in range(1, 6):
        sufijo = "" if i == 1 else f"_{i}"
        archivo_salida = f"resultado_init_llm{sufijo}.json"
        
        print(f" -> Ejecutando corrida {i}/5: {archivo_salida}...")
        ejecutar_experimento(
            generaciones=10, 
            tamano_poblacion=40, 
            tamano_refset=10, 
            archivo_salida=archivo_salida
        )

    tiempo_total = time.time() - tiempo_global_inicio
    print(f"\nBatería Fase 1 completada en {tiempo_total/60:.2f} minutos.")

if __name__ == "__main__":
    ejecutar_bateria_fase1()