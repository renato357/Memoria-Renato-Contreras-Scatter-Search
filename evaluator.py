import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

class EvaluadorMetricas:
    def __init__(self, sbert_model_name='all-MiniLM-L6-v2'):
        print(f"Cargando modelo SBERT: {sbert_model_name}...")
        self.sbert_model = SentenceTransformer(sbert_model_name)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Descargando dependencias de NLTK (solo la primera vez)...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True) # <-- clave
            
        self.smoother = SmoothingFunction().method1

    def calcular_calidad_sbert(self, texto_generado: str, texto_referencia: str) -> float:
        """
        Calcula la similitud semántica (Calidad) entre el dato sintético y el dato real.
        Retorna un valor float, usualmente entre 0.0 (nada similar) y 1.0 (idénticos).
        """
        if not texto_generado or not texto_referencia:
            return 0.0
        
        # Generar embeddings de ambas frases
        emb_generado = self.sbert_model.encode(texto_generado, convert_to_tensor=True)
        emb_referencia = self.sbert_model.encode(texto_referencia, convert_to_tensor=True)
        
        # Calcular similitud del coseno
        similitud = util.cos_sim(emb_generado, emb_referencia)
        return similitud.item()

    def calcular_diversidad_bleu(self, texto_candidato: str, textos_elite: list) -> float:
        if not texto_candidato or not textos_elite:
            return 0.0

        # Tokenizamos el candidato (separamos por palabras)
        candidato_tokens = nltk.word_tokenize(texto_candidato.lower())
        
        # Tokenizamos todo el grupo de referencia (la élite del RefSet)
        referencias_tokens = [nltk.word_tokenize(ref.lower()) for ref in textos_elite]
        
        # Calculamos BLEU enfocado en unigramas y bigramas (weights=0.5, 0.5)
        # Esto es ideal para tweets, ya que son textos cortos.
        score_bleu = sentence_bleu(
            referencias_tokens, 
            candidato_tokens, 
            weights=(0.5, 0.5, 0, 0), 
            smoothing_function=self.smoother
        )
        
        return score_bleu

# --- PRUEBA RÁPIDA DE CONCEPTO ---
if __name__ == "__main__":
    evaluador = EvaluadorMetricas()
    
    tweet_real = "The earthquake in Chile was terrifying, many buildings collapsed."
    
    tweet_sintetico_bueno = "A massive earthquake struck Chile today causing severe damage to structures."
    tweet_sintetico_malo = "I really like to eat pizza on fridays."
    
    print("\n--- PRUEBA DE CALIDAD (SBERT) ---")
    calidad_buena = evaluador.calcular_calidad_sbert(tweet_sintetico_bueno, tweet_real)
    calidad_mala = evaluador.calcular_calidad_sbert(tweet_sintetico_malo, tweet_real)
    print(f"Calidad del tweet bueno: {calidad_buena:.4f} (Debería ser alta)")
    print(f"Calidad del tweet malo:  {calidad_mala:.4f} (Debería ser baja)")
    
    print("\n--- PRUEBA DE DIVERSIDAD LÉXICA (BLEU) ---")
    elite_RefSet = [
        "A massive earthquake struck Chile today causing severe damage to structures.",
        "Major structural damage reported after heavy earthquake hit Chile."
    ]
    
    candidato_copion = "A massive earthquake hit Chile causing damage to structures."
    candidato_diverso = "Tremors shook the capital, leaving homes destroyed in South America."
    
    bleu_copion = evaluador.calcular_diversidad_bleu(candidato_copion, elite_RefSet)
    bleu_diverso = evaluador.calcular_diversidad_bleu(candidato_diverso, elite_RefSet)
    
    print(f"BLEU del candidato copión:  {bleu_copion:.4f} (Penaliza la diversidad)")
    print(f"BLEU del candidato diverso: {bleu_diverso:.4f} (Premia la diversidad)")