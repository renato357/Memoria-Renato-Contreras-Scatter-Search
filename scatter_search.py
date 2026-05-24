import random
import itertools
from structures import PromptSolution
from evaluator import EvaluadorMetricas
from llm import MotorLLM
from dataset import DatasetManager

class ScatterSearch:
    def __init__(self, tamano_poblacion=20, tamano_refset=10):
        self.P_size = tamano_poblacion
        self.b = tamano_refset
        self.b_elite = self.b // 2
        self.b_div = self.b - self.b_elite
        
        print("--- INICIALIZANDO SS-GrIPS ---")
        self.evaluador = EvaluadorMetricas()
        self.llm = MotorLLM()
        self.dataset = DatasetManager()
        
        # Cargamos una muestra de referencia (50 tweets reales para evaluar SBERT)
        self.textos_referencia = self.dataset.obtener_muestra_referencia(n=50, semilla=42)

    def generar_poblacion_inicial(self) -> list:
        """
        Genera P soluciones iniciales combinando 10 Roles y 10 Tareas (100 combinaciones).
        Mantenido por compatibilidad de la clase base.
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
                
        random.seed(42)
        random.shuffle(poblacion)
        return poblacion[:self.P_size]

    def generar_poblacion_inicial_llm(self) -> list:
        """
        [GANADOR FASE 1] Genera P soluciones iniciales usando a Llama 3 para aplicar 
        Ingeniería de Prompts Inversa basándose en ejemplos reales del dataset.
        """
        print(f"\n[Fase 1 - Configuración LLM] Generando población inicial de {self.P_size} individuos con Llama 3...")
        textos_ejemplo = self.dataset.obtener_muestra_referencia(n=self.P_size, semilla=99)
        poblacion = []
        
        for i, texto in enumerate(textos_ejemplo):
            print(f"  -> Generando prompt {i+1}/{self.P_size} basado en data real...", end="", flush=True)
            prompt_meta = (
                f"Read this real tweet: '{texto}'. "
                "Create a 'Role' and a 'Task' to instruct an AI to write a similar tweet. "
                "Reply ONLY in this exact format: 'Role: [your role]. Task: [your task].'"
            )
            
            try:
                respuesta = self.llm.invocar(prompt_meta, system_prompt="You are an expert prompt engineer.", temp=0.9)
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
        for ref in self.textos_referencia:
            sim = self.evaluador.calcular_calidad_sbert(texto_generado, ref)
            if sim > max_sim:
                max_sim = sim
                
        solucion.score_sbert = max_sim
        return solucion

    def construir_refset(self, poblacion: list) -> list:
        """
        Divide la población evaluada en b_elite (por SBERT) y b_div (por BLEU).
        """
        print(f"\n[Fase 2] Evaluando Calidad (SBERT) para {len(poblacion)} prompts...")
        for i, sol in enumerate(poblacion):
            print(f"  Evaluando prompt {i+1}/{len(poblacion)}...")
            self.evaluar_solucion(sol)
            
        poblacion.sort(key=lambda x: x.score_sbert, reverse=True)
        
        refset_elite = poblacion[:self.b_elite]
        restantes = poblacion[self.b_elite:]
        
        print(f"\n[Fase 2] Evaluando Diversidad Léxica (BLEU) para los {len(restantes)} restantes...")
        textos_tweets_elite = [sol.dato_generado for sol in refset_elite]
        
        for sol in restantes:
            sol.score_bleu = self.evaluador.calcular_diversidad_bleu(sol.dato_generado, textos_tweets_elite)
            
        restantes.sort(key=lambda x: x.score_bleu)
        refset_div = restantes[:self.b_div]
        
        refset_final = refset_elite + refset_div
        print("\n--- CONSTRUCCIÓN DEL REFSET COMPLETADA ---")
        return refset_final

    def generar_pares(self, refset: list) -> list:
        return list(itertools.combinations(refset, 2))

    def combinar_soluciones(self, p1: PromptSolution, p2: PromptSolution) -> list:
        """
        [VERSIÓN PURA] Aplica los operadores de cruce estructurados de forma mecánica.
        """
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
        """
        [VARIANTE COHERENCIA] Cruza las soluciones, pero llama a Llama 3 para reparar 
        la gramática y sintaxis de los hijos del Crossover Semántico antes de evaluarlos.
        """
        hijos = []
        
        # 1. Role Swapping (Intercambio de roles - Estructuralmente ya es coherente)
        hijo1 = PromptSolution(rol=p1.rol, tarea=p2.tarea, origen="role_swapping")
        hijo2 = PromptSolution(rol=p2.rol, tarea=p1.tarea, origen="role_swapping")
        hijos.extend([hijo1, hijo2])
        
        # 2. Semantic Task Crossover (Cruce a nivel de token)
        t1_tokens = p1.tarea.split()
        t2_tokens = p2.tarea.split()
        
        if len(t1_tokens) >= 2 and len(t2_tokens) >= 2:
            mid1 = len(t1_tokens) // 2
            mid2 = len(t2_tokens) // 2
            
            tarea_mixta_1 = " ".join(t1_tokens[:mid1] + t2_tokens[mid2:])
            tarea_mixta_2 = " ".join(t2_tokens[:mid2] + t1_tokens[mid1:])
            
            # Corrección Gramatical con LLM para el Hijo 3
            prompt_corr1 = (
                f"Fix the grammar and punctuation of this prompt task so it reads naturally, "
                f"without changing its core objective or constraints: '{tarea_mixta_1}'. "
                f"Reply ONLY with the corrected sentence, no quotes or notes."
            )
            try:
                tarea_corr1 = self.llm.invocar(prompt_corr1, system_prompt="You are a strict text editor.", temp=0.2)
                tarea_corr1 = tarea_corr1.replace('"', '').replace("'", "").strip()
                if tarea_corr1:
                    tarea_mixta_1 = tarea_corr1
            except Exception:
                pass

            # Corrección Gramatical con LLM para el Hijo 4
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