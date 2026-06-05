import time
from main import ejecutar_experimento
from scatter_search import ScatterSearch

def ejecutar_prueba_estrategias():
    print("==================================================")
    print(" INICIANDO TEST A/B: TABLERO DE CONTROL MAESTRO")
    print("==================================================\n")

    # ==========================================
    # ⚙️ TABLERO DE PARÁMETROS GLOBALES
    # Cambia estos valores antes de cada corrida
    # ==========================================
    PARAM_GENERACIONES = 10
    PARAM_POBLACION = 40
    PARAM_REFSET = 10
    
    # Fija un número (ej. 42) para reproducibilidad científica, o pon None para aleatorio total
    PARAM_SEMILLA = 42 
    
    MODELO_LLM = "llama3.1:8b"
    MODELO_SBERT = "all-MiniLM-L6-v2"
    MAX_TOKENS = 150
    # ==========================================
    
    tiempo_global_inicio = time.time()
    
    # Forzamos la inicialización LLM (esencial para probar validación cruzada con contexto real)
    ScatterSearch.generar_poblacion_inicial = ScatterSearch.generar_poblacion_inicial_llm
    
    # --- PRUEBA A: VALIDACIÓN ESTÁNDAR (False) ---
    print(">>> LANZANDO PRUEBA A: VALIDACIÓN ESTÁNDAR (Sin exclusión de textos de contexto)")
    ejecutar_experimento(
        generaciones=PARAM_GENERACIONES, 
        tamano_poblacion=PARAM_POBLACION, 
        tamano_refset=PARAM_REFSET, 
        archivo_salida="resultado_eval_estandar.json",
        validacion_cruzada=False,
        llm_model_name=MODELO_LLM,
        sbert_model_name=MODELO_SBERT,
        max_tokens_salida=MAX_TOKENS,
        semilla_global=PARAM_SEMILLA
    )
    
    print("\n[PAUSA TÉCNICA] Dejando enfriar el equipo por 30 segundos...\n")
    time.sleep(30)
    
    # --- PRUEBA B: VALIDACIÓN CRUZADA (True) ---
    print(">>> LANZANDO PRUEBA B: VALIDACIÓN CRUZADA (Gold Standard 100% excluyente)")
    ejecutar_experimento(
        generaciones=PARAM_GENERACIONES, 
        tamano_poblacion=PARAM_POBLACION, 
        tamano_refset=PARAM_REFSET, 
        archivo_salida="resultado_eval_cruzada.json",
        validacion_cruzada=True,
        llm_model_name=MODELO_LLM,
        sbert_model_name=MODELO_SBERT,
        max_tokens_salida=MAX_TOKENS,
        semilla_global=PARAM_SEMILLA
    )
    
    tiempo_total = time.time() - tiempo_global_inicio
    print(f"\n✅ Batería de Pruebas completada exitosamente en {tiempo_total/60:.2f} minutos.")

if __name__ == "__main__":
    ejecutar_prueba_estrategias()