import json
import time
from main import ejecutar_experimento
from scatter_search import ScatterSearch

# ==========================================
# ⚙️ TABLERO DE PARÁMETROS GLOBALES
# ==========================================
MODELO_LLM = "llama3.1:8b"
MODELO_SBERT = "all-MiniLM-L6-v2"
MAX_TOKENS = 150

def experimento_1_base():
    print("\n" + "="*50)
    print(" INICIANDO EXPERIMENTO 1: EFICIENCIA BASE")
    print("="*50)
    
    # G=10, P=40, R=10, 5 corridas
    resultados = []
    
    for i in range(1, 6):
        print(f"\n>> Corriendo Exp 1 - Ejecución {i}/5")
        salida = ejecutar_experimento(
            generaciones=10,
            tamano_poblacion=40,
            tamano_refset=10,
            archivo_salida=f"temp_run_exp1.json", 
            validacion_cruzada=False,
            llm_model_name=MODELO_LLM,
            sbert_model_name=MODELO_SBERT,
            max_tokens_salida=MAX_TOKENS,
            semilla_global=None # None para garantizar independencia estocástica
        )
        resultados.append(salida)
        
    with open("exp1_base.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 1 Finalizado. Resultados guardados en exp1_base.json")


def experimento_2_sensibilidad():
    print("\n" + "="*50)
    print(" INICIANDO EXPERIMENTO 2: SENSIBILIDAD DE HIPERPARÁMETROS")
    print("="*50)
    
    resultados_totales = {}
    
    # FASE A: Fijar Población=40, Variar RefSet
    poblacion_fija = 40
    refsets_a_probar = [5, 10, 15]
    
    for r in refsets_a_probar:
        config_name = f"conf_P{poblacion_fija}_R{r}"
        if config_name not in resultados_totales:
            resultados_totales[config_name] = []
            
        for i in range(1, 4):
            print(f"\n>> Exp 2 (Fase A) - {config_name} - Ejecución {i}/3")
            salida = ejecutar_experimento(
                generaciones=10,
                tamano_poblacion=poblacion_fija,
                tamano_refset=r,
                archivo_salida="temp_run_exp2.json",
                validacion_cruzada=False,
                llm_model_name=MODELO_LLM,
                sbert_model_name=MODELO_SBERT,
                max_tokens_salida=MAX_TOKENS,
                semilla_global=None
            )
            resultados_totales[config_name].append(salida)
            
    # FASE B: Fijar RefSet=10, Variar Población
    refset_fijo = 10
    poblaciones_a_probar = [20, 40, 60]
    
    for p in poblaciones_a_probar:
        config_name = f"conf_P{p}_R{refset_fijo}"
        if config_name not in resultados_totales:
            resultados_totales[config_name] = []
            
        for i in range(1, 4):
            print(f"\n>> Exp 2 (Fase B) - {config_name} - Ejecución {i}/3")
            salida = ejecutar_experimento(
                generaciones=10,
                tamano_poblacion=p,
                tamano_refset=refset_fijo,
                archivo_salida="temp_run_exp2.json",
                validacion_cruzada=False,
                llm_model_name=MODELO_LLM,
                sbert_model_name=MODELO_SBERT,
                max_tokens_salida=MAX_TOKENS,
                semilla_global=None
            )
            resultados_totales[config_name].append(salida)
            
    with open("exp2_sensibilidad.json", "w", encoding="utf-8") as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 2 Finalizado. Resultados guardados en exp2_sensibilidad.json")


def experimento_3_radiografia():
    print("\n" + "="*50)
    print(" INICIANDO EXPERIMENTO 3: RADIOGRAFÍA GrIPS")
    print("="*50)
    
    # G=10, P=40, R=10, 3 corridas
    resultados = []
    
    for i in range(1, 4):
        print(f"\n>> Corriendo Exp 3 - Ejecución {i}/3")
        salida = ejecutar_experimento(
            generaciones=10,
            tamano_poblacion=40,
            tamano_refset=10,
            archivo_salida="temp_run_exp3.json",
            validacion_cruzada=False,
            llm_model_name=MODELO_LLM,
            sbert_model_name=MODELO_SBERT,
            max_tokens_salida=MAX_TOKENS,
            semilla_global=None
        )
        resultados.append(salida)
        
    with open("exp3_radiografia.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 3 Finalizado. Resultados guardados en exp3_radiografia.json")


if __name__ == "__main__":
    # Forzamos la inicialización LLM para todas las pruebas
    ScatterSearch.generar_poblacion_inicial = ScatterSearch.generar_poblacion_inicial_llm
    
    tiempo_inicio = time.time()
    
    experimento_1_base()
    
    print("\n[PAUSA TÉCNICA] Enfriando sistema por 10 segundos...")
    time.sleep(10)
    
    experimento_2_sensibilidad()
    
    print("\n[PAUSA TÉCNICA] Enfriando sistema por 10 segundos...")
    time.sleep(10)
    
    experimento_3_radiografia()
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n🎉 TODA LA BATERÍA DE EXPERIMENTOS COMPLETADA EN {tiempo_total/60:.2f} MINUTOS.")
    print("Ya puedes revisar exp1_base.json, exp2_sensibilidad.json y exp3_radiografia.json.")