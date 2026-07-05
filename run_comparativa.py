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
            validacion_cruzada=True,
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
    print(" INICIANDO EXPERIMENTO 2: SENSIBILIDAD (REFSET Y GENERACIONES)")
    print("="*50)
    
    resultados_totales = {}
    
    # FASE A: Fijar Generaciones=10, Población=40, Variar RefSet
    generacion_fija = 10
    poblacion_fija = 40
    refsets_a_probar = [5, 10, 15]
    
    for r in refsets_a_probar:
        config_name = f"conf_G{generacion_fija}_R{r}"
        if config_name not in resultados_totales:
            resultados_totales[config_name] = []
            
        for i in range(1, 4):
            print(f"\n>> Exp 2 (Fase A) - {config_name} - Ejecución {i}/3")
            salida = ejecutar_experimento(
                generaciones=generacion_fija,
                tamano_poblacion=poblacion_fija,
                tamano_refset=r,
                archivo_salida="temp_run_exp2.json",
                validacion_cruzada=True,
                llm_model_name=MODELO_LLM,
                sbert_model_name=MODELO_SBERT,
                max_tokens_salida=MAX_TOKENS,
                semilla_global=None
            )
            resultados_totales[config_name].append(salida)
            
    # FASE B: Fijar Población=40, RefSet=10, Variar Generaciones
    refset_fijo = 10
    generaciones_a_probar = [5, 15] # G10 ya se hizo en Fase A
    
    for g in generaciones_a_probar:
        config_name = f"conf_G{g}_R{refset_fijo}"
        if config_name not in resultados_totales:
            resultados_totales[config_name] = []
            
        for i in range(1, 4):
            print(f"\n>> Exp 2 (Fase B) - {config_name} - Ejecución {i}/3")
            salida = ejecutar_experimento(
                generaciones=g,
                tamano_poblacion=poblacion_fija,
                tamano_refset=refset_fijo,
                archivo_salida="temp_run_exp2.json",
                validacion_cruzada=True,
                llm_model_name=MODELO_LLM,
                sbert_model_name=MODELO_SBERT,
                max_tokens_salida=MAX_TOKENS,
                semilla_global=None
            )
            resultados_totales[config_name].append(salida)
            
    with open("exp2_sensibilidad.json", "w", encoding="utf-8") as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 2 Finalizado. Resultados guardados en exp2_sensibilidad.json")


def experimento_3_data_leakage():
    print("\n" + "="*50)
    print(" INICIANDO EXPERIMENTO 3: DATA LEAKAGE (CRUZADA OFF)")
    print("="*50)
    
    # G=10, P=40, R=10, 3 corridas
    resultados = []
    
    for i in range(1, 4):
        print(f"\n>> Corriendo Exp 3 (Cruzada OFF) - Ejecución {i}/3")
        salida = ejecutar_experimento(
            generaciones=10,
            tamano_poblacion=40,
            tamano_refset=10,
            archivo_salida="temp_run_exp3.json",
            validacion_cruzada=False,  # ESTO ESTÁ APAGADO PARA PROBAR DATA LEAKAGE
            llm_model_name=MODELO_LLM,
            sbert_model_name=MODELO_SBERT,
            max_tokens_salida=MAX_TOKENS,
            semilla_global=None
        )
        resultados.append(salida)
        
    with open("exp3_cruzada_off.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 3 Finalizado. Resultados guardados en exp3_cruzada_off.json")


def experimento_4_temperaturas():
    print("\n" + "="*50)
    print(" INICIANDO EXPERIMENTO 4: IMPACTO DE LA TEMPERATURA EN GrIPS")
    print("="*50)
    
    resultados_totales = {}
    temperaturas = [0.0, 0.4, 0.6]
    
    for temp in temperaturas:
        config_name = f"conf_T{temp}"
        resultados_totales[config_name] = []
        
        for i in range(1, 4):
            print(f"\n>> Corriendo Exp 4 (Temp={temp}) - Ejecución {i}/3")
            salida = ejecutar_experimento(
                generaciones=10,
                tamano_poblacion=40,
                tamano_refset=10,
                archivo_salida=f"temp_run_exp4.json",
                validacion_cruzada=True,
                llm_model_name=MODELO_LLM,
                sbert_model_name=MODELO_SBERT,
                max_tokens_salida=MAX_TOKENS,
                semilla_global=None,
                temp_paraphrase=temp
            )
            resultados_totales[config_name].append(salida)
            
    with open("exp4_temperaturas.json", "w", encoding="utf-8") as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 4 Finalizado. Resultados guardados en exp4_temperaturas.json")

def experimento_5_inicializacion():
    print("\n" + "="*50)
    print(" INICIANDO EXPERIMENTO 5: TEMPERATURA EN INICIALIZACIÓN")
    print("="*50)
    
    resultados_totales = {}
    temperaturas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for temp in temperaturas:
        config_name = f"conf_T{temp}"
        resultados_totales[config_name] = []
        
        for i in range(1, 4):
            print(f"\n>> Corriendo Exp 5 (Temp={temp}) - Ejecución {i}/3")
            tiempo_inicio = time.time()
            
            ss = ScatterSearch(
                tamano_poblacion=40,
                tamano_refset=10,
                validacion_cruzada=True,
                llm_model_name=MODELO_LLM,
                sbert_model_name=MODELO_SBERT,
                max_tokens_salida=MAX_TOKENS,
                semilla_global=None,
                temp_inicial=temp
            )
            
            poblacion = ss.generar_poblacion_inicial_llm()
            
            sberts = []
            fallbacks = 0
            palabras_totales = set()
            
            print(f"  Evaluando {len(poblacion)} prompts iniciales...")
            for sol in poblacion:
                if sol.origen == "init_llm_fallback":
                    fallbacks += 1
                
                # Evaluar prompt
                texto_gen = ss.llm.invocar(sol.prompt_completo)
                sol.dato_generado = texto_gen
                
                max_sbert = 0.0
                for ref in ss.textos_referencia:
                    sim = ss.evaluador.calcular_calidad_sbert(texto_gen, ref)
                    if sim > max_sbert:
                        max_sbert = sim
                sberts.append(max_sbert)
                
                # Diversidad léxica basada en el PROMPT generado (Role/Task)
                palabras_prompt = sol.prompt_completo.lower().split()
                palabras_totales.update(palabras_prompt)
                
            tiempo_fin = time.time()
            
            salida = {
                "temperatura": temp,
                "corrida": i,
                "sbert_maximo": float(max(sberts)) if sberts else 0.0,
                "sbert_promedio": float(sum(sberts)/len(sberts)) if sberts else 0.0,
                "fallbacks": fallbacks,
                "diversidad_lexica": len(palabras_totales),
                "tiempo_minutos": (tiempo_fin - tiempo_inicio) / 60.0,
                "prompts_generados": [{"rol": s.rol, "tarea": s.tarea, "origen": s.origen, "sbert": float(sberts[idx])} for idx, s in enumerate(poblacion)]
            }
            
            resultados_totales[config_name].append(salida)
            
    with open("exp5_inicializacion.json", "w", encoding="utf-8") as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)
    print("\n✅ Experimento 5 Finalizado. Resultados guardados en exp5_inicializacion.json")

if __name__ == "__main__":
    # Forzamos la inicialización LLM para todas las pruebas
    ScatterSearch.generar_poblacion_inicial = ScatterSearch.generar_poblacion_inicial_llm
    
    tiempo_inicio = time.time()
    
    # experimento_1_base()
    # experimento_2_sensibilidad()
    # experimento_3_data_leakage()
    # experimento_4_temperaturas()
    
    experimento_5_inicializacion()
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n🎉 TODA LA BATERÍA DE EXPERIMENTOS COMPLETADA EN {tiempo_total/60:.2f} MINUTOS.")
    print("Revisar resultados en exp5_inicializacion.json.")