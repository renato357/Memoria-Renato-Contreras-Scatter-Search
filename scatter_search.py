import random
import itertools
from structures import PromptSolution
from evaluator import EvaluadorMetricas
from llm import MotorLLM
from dataset import DatasetManager

class ScatterSearch:
    def __init__(self, 
                 tamano_poblacion=40, 
                 tamano_refset=10, 
                 validacion_cruzada=True,
                 llm_model_name="llama3.1:8b",
                 sbert_model_name="all-MiniLM-L6-v2",
                 max_tokens_salida=150,
                 semilla_global=None,
                 temp_inicial=0.9):
        """
        Inicializa el algoritmo recibiendo todos los hiperparámetros desde la función principal.
        """
        self.P_size = tamano_poblacion
        self.temp_inicial = temp_inicial
        self.b = tamano_refset
        self.b_elite = self.b // 2
        self.b_div = self.b - self.b_elite
        self.validacion_cruzada = validacion_cruzada
        self.semilla_global = semilla_global
                    
        print("--- INICIALIZANDO SS-GrIPS ---")
        
        # Inyectamos los parámetros controlados a los módulos secundarios
        self.evaluador = EvaluadorMetricas(sbert_model_name=sbert_model_name)
        self.llm = MotorLLM(
            model_name=llm_model_name, 
            semilla=self.semilla_global, 
            max_tokens_salida=max_tokens_salida
        )
        self.dataset = DatasetManager()
        
        # 1. Obtenemos los textos que usará Llama 3 para inspirarse (Fase 1)
        # Ya NO pasamos la semilla. Usará la secuencia global establecida en main.py
        self.textos_contexto_llm = self.dataset.obtener_muestra_referencia(
            n=self.P_size
        )
        
        # 2. Obtenemos el "Gold Standard" de evaluación (SBERT) congelado para todo el ciclo
        if self.validacion_cruzada:
            print("[Estrategia] Validación Cruzada: El Gold Standard será excluyente.")
            self.textos_referencia = self.dataset.obtener_muestra_referencia(
                n=50, 
                excluir_textos=self.textos_contexto_llm
            )
        else:
            print("[Estrategia] Validación Estándar (Data Leakage Total): El Gold Standard será EXACTAMENTE los textos usados para inspirar.")
            self.textos_referencia = self.textos_contexto_llm.copy()

    def generar_poblacion_inicial(self) -> list:
        """
        Genera P soluciones iniciales de forma determinista (Baseline 10x10).
        """
        roles = [
            "You are a public health official issuing a formal statement.",
            "Act as a frontline doctor sharing a critical update from the hospital.",
            "You are an alarmed citizen tweeting about local lockdown measures.",
            "Act as a skeptical individual questioning the new safety guidelines.",
            "You are a data journalist reporting the latest infection statistics.",
            "Act as a frustrated small business owner dealing with pandemic closures.",
            "You are a scientific researcher explaining a new virus variant.",
            "Act as a compassionate neighbor offering community support during isolation.",
            "You are a government spokesperson announcing a public health campaign.",
            "Act as a panicked parent worried about school reopenings."
        ]
        
        tareas = [
            "Write a short tweet about the impact of the COVID-19 pandemic on daily life.",
            "Draft a Twitter post urging people to follow social distancing rules.",
            "Create a brief update regarding the shortage of medical supplies.",
            "Write a short message reacting to the latest coronavirus news.",
            "Generate a social media post asking for help during the quarantine.",
            "Draft a tweet expressing hope and resilience against the virus.",
            "Write a short warning about the rapid spread of COVID-19 in the city.",
            "Create a brief Twitter announcement about local safety protocols.",
            "Generate a message discussing the economic consequences of the lockdown.",
            "Draft a short post debunking a common myth about the pandemic."
        ]
        
        poblacion = []
        for r in roles:
            for t in tareas:
                poblacion.append(PromptSolution(rol=r, tarea=t, origen="inicializacion"))
                
        random.shuffle(poblacion)
        return poblacion[:self.P_size]

    def generar_poblacion_inicial_llm(self) -> list:
        """
        [LLM Init] Genera P soluciones iniciales basándose en el contexto extraído.
        """
        print(f"\n[Fase 1] Generando población inicial de {self.P_size} individuos con LLM...")
        poblacion = []
        
        for i, texto in enumerate(self.textos_contexto_llm):
            print(f"  -> Generando prompt {i+1}/{self.P_size} basado en data real...", end="", flush=True)
            prompt_meta = (
                f"Read this real tweet: '{texto}'. "
                "Create a 'Role' and a 'Task' to instruct an AI to write a similar tweet. "
                "Reply ONLY in this exact format: 'Role: [your role]. Task: [your task].'"
            )
            
            try:
                # Usamos la temperatura de inicialización configurada
                respuesta = self.llm.invocar(prompt_meta, system_prompt="You are an expert prompt engineer.", temp=self.temp_inicial)
                if "Role:" in respuesta and "Task:" in respuesta:
                    partes = respuesta.split("Task:")
                    rol = partes[0].replace("Role:", "").strip()
                    tarea = partes[1].strip()
                    poblacion.append(PromptSolution(rol=rol, tarea=tarea, origen="init_llm"))
                    print(" ¡Listo!")
                else:
                    poblacion.append(PromptSolution(rol="You are a social media user.", tarea=f"Write a short tweet similar in tone to: {texto[:30]}...", origen="init_llm_fallback"))
                    print(" (Fallback usado)")
            except Exception:
                 poblacion.append(PromptSolution(rol="You are a citizen.", tarea="Write a short tweet about the pandemic.", origen="init_llm_error"))
                 print(" (Error LLM)")
                 
        return poblacion

    def evaluar_solucion(self, solucion: PromptSolution) -> PromptSolution:
        texto_generado = self.llm.invocar(solucion.prompt_completo)
        solucion.dato_generado = texto_generado
        
        max_sim = 0.0
        mejor_ref = ""
        for ref in self.textos_referencia:
            sim = self.evaluador.calcular_calidad_sbert(texto_generado, ref)
            if sim > max_sim:
                max_sim = sim
                mejor_ref = ref # Guardamos contra qué texto hizo match
                
        solucion.score_sbert = max_sim
        solucion.texto_referencia_match = mejor_ref
        return solucion
                
        solucion.score_sbert = max_sim
        return solucion

    def construir_refset(self, poblacion: list) -> list:
        """
        Mantiene un conjunto de referencia (RefSet) puramente elitista
        basado estrictamente en la métrica de calidad semántica (SBERT).
        """
        print(f"\n[Fase 2] Evaluando Calidad (SBERT) para {len(poblacion)} prompts...")
        for i, sol in enumerate(poblacion):
            # Optimización: Solo invoca al LLM y calcula SBERT si es un individuo nuevo
            if sol.score_sbert == 0.0:
                print(f"  Evaluando prompt {i+1}/{len(poblacion)}...")
                self.evaluar_solucion(sol)
            
        # Ordenamos la población entera de mayor a menor calidad (SBERT)
        poblacion.sort(key=lambda x: x.score_sbert, reverse=True)
        
        # Selección Elitista con Niching (Bloqueo de Convergencia a 1 Tweet)
        refset_final = []
        conteo_objetivos = {}
        max_por_objetivo = 2
        
        for sol in poblacion:
            objetivo = sol.texto_referencia_match
            if conteo_objetivos.get(objetivo, 0) < max_por_objetivo:
                refset_final.append(sol)
                conteo_objetivos[objetivo] = conteo_objetivos.get(objetivo, 0) + 1
            if len(refset_final) >= self.b:
                break
        
        print("\n--- CONSTRUCCIÓN DEL REFSET (NICHING) COMPLETADA ---")
        return refset_final

    def generar_pares(self, refset: list) -> list:
        return list(itertools.combinations(refset, 2))

    def combinar_soluciones(self, p1: PromptSolution, p2: PromptSolution) -> list:
        hijos = []
        hijo1 = PromptSolution(rol=p1.rol, tarea=p2.tarea, origen="role_swapping")
        hijo2 = PromptSolution(rol=p2.rol, tarea=p1.tarea, origen="role_swapping")
        hijos.extend([hijo1, hijo2])
        
        t1_tokens = p1.tarea.split()
        t2_tokens = p2.tarea.split()
        
        if len(t1_tokens) >= 2 and len(t2_tokens) >= 2:
            mid1 = len(t1_tokens) // 2
            mid2 = len(t2_tokens) // 2
            
            tarea_mixta_1 = " ".join(t1_tokens[:mid1] + t2_tokens[mid2:])
            tarea_mixta_2 = " ".join(t2_tokens[:mid2] + t1_tokens[mid1:])
            
            hijo3 = PromptSolution(rol=p1.rol, tarea=tarea_mixta_1, origen="semantic_crossover")
            hijo4 = PromptSolution(rol=p2.rol, tarea=tarea_mixta_2, origen="semantic_crossover")
            hijos.extend([hijo3, hijo4])
            
        return hijos

    def combinar_soluciones_coherente(self, p1: PromptSolution, p2: PromptSolution) -> list:
        hijos = []
        
        hijo1 = PromptSolution(rol=p1.rol, tarea=p2.tarea, origen="role_swapping")
        hijo2 = PromptSolution(rol=p2.rol, tarea=p1.tarea, origen="role_swapping")
        hijos.extend([hijo1, hijo2])
        
        t1_tokens = p1.tarea.split()
        t2_tokens = p2.tarea.split()
        
        if len(t1_tokens) >= 2 and len(t2_tokens) >= 2:
            mid1 = len(t1_tokens) // 2
            mid2 = len(t2_tokens) // 2
            
            tarea_mixta_1 = " ".join(t1_tokens[:mid1] + t2_tokens[mid2:])
            tarea_mixta_2 = " ".join(t2_tokens[:mid2] + t1_tokens[mid1:])
            
            prompt_corr1 = (
                f"Fix the grammar and punctuation of this prompt task so it reads naturally, "
                f"without changing its core objective or constraints: '{tarea_mixta_1}'. "
                f"Reply ONLY with the corrected sentence, no quotes or notes."
            )
            try:
                # Temperatura de 0.2 para corregir gramática sin alucinar contenido extra
                tarea_corr1 = self.llm.invocar(prompt_corr1, system_prompt="You are a strict text editor.", temp=0.2)
                tarea_corr1 = tarea_corr1.replace('"', '').replace("'", "").strip()
                if tarea_corr1:
                    tarea_mixta_1 = tarea_corr1
            except Exception:
                pass

            prompt_corr2 = (
                f"Fix the grammar and punctuation of this prompt task so it reads naturally, "
                f"without changing its core objective or constraints: '{tarea_mixta_2}'. "
                f"Reply ONLY with the corrected sentence, no quotes or notes."
            )
            try:
                tarea_corr2 = self.llm.invocar(prompt_corr2, system_prompt="You are a strict text editor.", temp=0.2)
                tarea_corr2 = tarea_corr2.replace('"', '').replace("'", "").strip()
                if tarea_corr2:
                    tarea_mixta_2 = tarea_corr2
            except Exception:
                pass
            
            hijo3 = PromptSolution(rol=p1.rol, tarea=tarea_mixta_1, origen="semantic_crossover_corrected")
            hijo4 = PromptSolution(rol=p2.rol, tarea=tarea_mixta_2, origen="semantic_crossover_corrected")
            hijos.extend([hijo3, hijo4])
            
        return hijos