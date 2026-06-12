import random
import re
from structures import PromptSolution
from llm import MotorLLM
from evaluator import EvaluadorMetricas

class ModuloGrIPS:
    def __init__(self, motor_llm: MotorLLM, evaluador: EvaluadorMetricas):
        """
        Inicializa el módulo de mejora local (Intensificación).
        """
        self.llm = motor_llm
        self.evaluador = evaluador
        
        # Stop words básicas para evitar borrar palabras clave
        self.stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "is", "with", "that"}
        
        # Estadísticas permanentes de GrIPS
        self.stats_operadores = {
            "grips_delete": {"intentos": 0, "exitos": 0, "mejora_acumulada": 0.0},
            "grips_swap": {"intentos": 0, "exitos": 0, "mejora_acumulada": 0.0},
            "grips_paraphrase": {"intentos": 0, "exitos": 0, "mejora_acumulada": 0.0},
            "grips_add": {"intentos": 0, "exitos": 0, "mejora_acumulada": 0.0}
        }

    def _dividir_en_frases(self, texto: str) -> list:
        """Divide el texto en partes usando signos de puntuación o conectores."""
        # Se corta por coma, punto, o conectores específicos
        partes = re.split(r'([,.]\s|\s+and\s+|\s+or\s+|\s+but\s+)', texto)
        # Filtramos para no dejar espacios vacíos
        return [p.strip() for p in partes if p.strip() and p.strip() not in [',', '.', 'and', 'or', 'but']]

    def aplicar_operadores(self, individuo: PromptSolution) -> list:
        """
        Aplica los 4 operadores de GrIPS (Delete, Swap, Paraphrase, Add) sobre la TAREA del prompt.
        Retorna una lista de candidatos (tuplas: (operador, nueva_tarea, frase_borrada_opcional)).
        """
        candidatos = []
        tarea_original = individuo.tarea
        frases = self._dividir_en_frases(tarea_original)
        
        # Si la frase es muy corta, la separamos por palabras para que GrIPS pueda operar
        if len(frases) < 2:
            frases = tarea_original.split()

        # 1. DELETE (Borrar) - CON CANDADO ANTI-REWARD HACKING
        if len(frases) > 1:
            idx_borrar = random.randint(0, len(frases) - 1)
            frase_borrada = frases[idx_borrar]
            
            # Solo borramos si no es una stop word suelta
            if len(frase_borrada) > 3 or frase_borrada.lower() not in self.stop_words:
                nuevas_frases = frases[:idx_borrar] + frases[idx_borrar+1:]
                tarea_resultante = " ".join(nuevas_frases)
                
                # --- LÓGICA DEL CANDADO DINÁMICO ---
                palabras_originales = len(tarea_original.split())
                palabras_resultantes = len(tarea_resultante.split())
                
                # Límite: Mínimo 3 palabras ABSOLUTAS, o el 50% del largo original
                limite_palabras = max(3, int(palabras_originales * 0.5))
                
                if palabras_resultantes >= limite_palabras:
                    candidatos.append(("grips_delete", tarea_resultante, frase_borrada))

        # 2. SWAP (Intercambiar)
        if len(frases) >= 2:
            idx1, idx2 = random.sample(range(len(frases)), 2)
            nuevas_frases = frases.copy()
            nuevas_frases[idx1], nuevas_frases[idx2] = nuevas_frases[idx2], nuevas_frases[idx1]
            candidatos.append(("grips_swap", " ".join(nuevas_frases), None))

        # 3. PARAPHRASE (Parafrasear - Usa el LLM muy rápido)
        if frases:
            idx_paraf = random.randint(0, len(frases) - 1)
            objetivo = frases[idx_paraf]
            # Le pedimos al LLM que parafrasee solo esa porción, sin evaluar SBERT aún
            prompt_paraf = f"Rewrite this phrase briefly: '{objetivo}'. Reply ONLY with the rewritten phrase, no quotes."
            
            try:
                # Usamos una temperatura baja (0.2) para que el parafraseo sea directo y no alucine
                nueva_frase = self.llm.invocar(prompt_paraf, system_prompt="You are a strict text editor.", temp=0.2)
                nueva_frase = nueva_frase.replace('"', '').replace("'", "").strip()
                
                if nueva_frase and nueva_frase.lower() != objetivo.lower():
                    nuevas_frases = frases.copy()
                    nuevas_frases[idx_paraf] = nueva_frase
                    candidatos.append(("grips_paraphrase", " ".join(nuevas_frases), None))
            except Exception:
                pass # Si falla el LLM, ignoramos el parafraseo y seguimos

        # 4. ADD (Añadir)
        if individuo.frases_borradas and frases:
            frase_a_anadir = random.choice(individuo.frases_borradas)
            insert_idx = random.randint(0, len(frases))
            nuevas_frases = frases[:insert_idx] + [frase_a_anadir] + frases[insert_idx:]
            candidatos.append(("grips_add", " ".join(nuevas_frases), None))

        return candidatos

    def ejecutar_greedy(self, individuo: PromptSolution, textos_referencia: list) -> PromptSolution:
        """
        Ejecuta GrIPS en modo Greedy con logs detallados de consola.
        """
        # 1. Calculamos el SBERT base (El rival a vencer)
        if individuo.score_sbert == 0.0:
            texto_base = self.llm.invocar(individuo.prompt_completo)
            max_sim_base = 0.0
            mejor_ref_base = ""
            for ref in textos_referencia:
                sim = self.evaluador.calcular_calidad_sbert(texto_base, ref)
                if sim > max_sim_base:
                    max_sim_base = sim
                    mejor_ref_base = ref
                    
            individuo.score_sbert = max_sim_base
            individuo.dato_generado = texto_base
            individuo.texto_referencia_match = mejor_ref_base # <-- El eslabón perdido

        print(f"  [GrIPS] SBERT Base a batir: {individuo.score_sbert:.4f}")

        # 2. Generamos todas las posibles mutaciones
        candidatos_mutados = self.aplicar_operadores(individuo)
        print(f"  [GrIPS] Se generaron {len(candidatos_mutados)} mutaciones posibles. Probando en modo Greedy...")
        
        mejor_individuo = individuo 

        # 3. Probamos una por una hasta que una gane
        for i, (operacion, nueva_tarea, borrado) in enumerate(candidatos_mutados):
            prompt_temp = f"Role: {individuo.rol}. Task: {nueva_tarea}"
            print(f"    -> Testeando mutación {i+1}/{len(candidatos_mutados)} ({operacion})... ", end="", flush=True)
            
            dato_prueba = self.llm.invocar(prompt_temp)
            
            # Buscar el max_sim y además guardar el texto real que hizo match
            max_sim_prueba = 0.0
            mejor_ref_prueba = ""
            for ref in textos_referencia:
                sim = self.evaluador.calcular_calidad_sbert(dato_prueba, ref)
                if sim > max_sim_prueba:
                    max_sim_prueba = sim
                    mejor_ref_prueba = ref
            
            # Registrar el intento
            if operacion in self.stats_operadores:
                self.stats_operadores[operacion]["intentos"] += 1
                
            if max_sim_prueba > mejor_individuo.score_sbert:
                print(f"¡ÉXITO! SBERT subió a {max_sim_prueba:.4f}")
                
                # Registrar el éxito y la mejora
                if operacion in self.stats_operadores:
                    mejora = max_sim_prueba - mejor_individuo.score_sbert
                    self.stats_operadores[operacion]["exitos"] += 1
                    self.stats_operadores[operacion]["mejora_acumulada"] += mejora
                    
                mejor_individuo = PromptSolution(
                    rol=individuo.rol,
                    tarea=nueva_tarea,
                    prompt_completo=prompt_temp,
                    dato_generado=dato_prueba,
                    texto_referencia_match=mejor_ref_prueba, # <--- NUEVO
                    score_sbert=max_sim_prueba,
                    origen=operacion
                )
                mejor_individuo.frases_borradas = individuo.frases_borradas.copy()
                if borrado:
                    mejor_individuo.frases_borradas.append(borrado)
                    
                return mejor_individuo 
            else:
                print(f"Falló (SBERT: {max_sim_prueba:.4f})")

        print("  [GrIPS] Ninguna mutación logró superar a la base. Se mantiene el prompt original.")
        return mejor_individuo

# --- PRUEBA AISLADA DEL MÓDULO GRIPS ---
if __name__ == "__main__":
    from dataset import DatasetManager
    
    print("Iniciando prueba de GrIPS...")
    motor = MotorLLM()
    evaluador = EvaluadorMetricas()
    ds = DatasetManager()
    referencias = ds.obtener_muestra_referencia(n=10) # Muestra pequeña para rapidez
    
    grips = ModuloGrIPS(motor, evaluador)
    
    # Creamos un individuo de prueba defectuoso a propósito
    hijo_prueba = PromptSolution(
        rol="You are a panicked citizen.",
        tarea="Write a short message for impact of the COVID-19 virus.", # Tarea cruzada rara del paso anterior
        origen="semantic_crossover"
    )
    
    print(f"\nEvaluando hijo base: {hijo_prueba.tarea}")
    hijo_mejorado = grips.ejecutar_greedy(hijo_prueba, referencias)
    
    print("\nRESULTADO GRIPS:")
    print(f"Origen final: {hijo_mejorado.origen}")
    print(f"Tarea final:  {hijo_mejorado.tarea}")
    print(f"SBERT:        {hijo_mejorado.score_sbert:.4f}")