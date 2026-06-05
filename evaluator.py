import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

class EvaluadorMetricas:
    def __init__(self, sbert_model_name='all-MiniLM-L6-v2'):
        """
        Inicializa los modelos de evaluación. 
        El modelo de SBERT ahora puede ser parametrizado desde el main.
        """
        print(f"Cargando modelo SBERT: {sbert_model_name}...")
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
        # Bloque seguro para descargar NLTK en servidores
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Descargando dependencias de NLTK (solo la primera vez)...")
            try:
                import ssl
                # Ignorar posibles errores de certificado SSL en redes universitarias
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                    
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                print(f"[Aviso] No se pudo descargar NLTK automáticamente. Error: {e}")
            
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
        """
        Calcula el BLEU enfocado en unigramas y bigramas. Ideal para textos cortos (tweets).
        """
        if not texto_candidato or not textos_elite:
            return 0.0

        candidato_tokens = nltk.word_tokenize(texto_candidato.lower())
        referencias_tokens = [nltk.word_tokenize(ref.lower()) for ref in textos_elite]
        
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
    
    print("\n--- PRUEBA DE CALIDAD (SBERT) ---")
    calidad_buena = evaluador.calcular_calidad_sbert(tweet_sintetico_bueno, tweet_real)
    print(f"Calidad del tweet sintético: {calidad_buena:.4f}")