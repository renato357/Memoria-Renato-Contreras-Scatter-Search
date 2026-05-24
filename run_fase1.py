import time
from main import ejecutar_experimento
from scatter_search import ScatterSearch

# Un pequeño "hack" (Monkey Patch) para cambiar temporalmente la inicialización de la clase SS
# sin ensuciar tu código original con variables IF complejas.

def ejecutar_bateria_fase1():
    print("==================================================")
    print(" INICIANDO TEST A/B: POBLACIÓN INICIAL")
    print("==================================================\n")
    
    tiempo_global_inicio = time.time()
    
    # --- PRUEBA A: EL BASELINE (10x10) ---
    print(">>> LANZANDO PRUEBA A: INICIALIZACIÓN DETERMINISTA (10x10)")
    ejecutar_experimento(
        generaciones=10, 
        tamano_poblacion=40, 
        tamano_refset=10, 
        archivo_salida="resultado_init_10x10.json"
    )
    
    print("\n[PAUSA TÉCNICA] Dejando enfriar el equipo por 30 segundos...\n")
    time.sleep(30)
    
    # --- PRUEBA B: EL LLM (Meta-Prompting) ---
    print(">>> LANZANDO PRUEBA B: INICIALIZACIÓN ESTOCÁSTICA (Llama 3)")
    
    # Hackeamos la clase ScatterSearch en tiempo de ejecución para que use la variante
    ScatterSearch.generar_poblacion_inicial = ScatterSearch.generar_poblacion_inicial_llm
    
    ejecutar_experimento(
        generaciones=10, 
        tamano_poblacion=40, 
        tamano_refset=10, 
        archivo_salida="resultado_init_llm.json"
    )
    
    tiempo_global_fin = (time.time() - tiempo_global_inicio) / 3600
    print("\n==================================================")
    print(f" BATERÍA FASE 1 COMPLETADA EN {tiempo_global_fin:.2f} HORAS.")
    print("==================================================")

if __name__ == "__main__":
    ejecutar_bateria_fase1()