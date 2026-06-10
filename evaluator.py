from sentence_transformers import SentenceTransformer, util

class EvaluadorMetricas:
    def __init__(self, sbert_model_name='all-MiniLM-L6-v2'):
        """
        Inicializa el modelo de evaluación de calidad semántica.
        """
        print(f"Cargando modelo SBERT: {sbert_model_name}...")
        self.sbert_model = SentenceTransformer(sbert_model_name)

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

# --- PRUEBA RÁPIDA DE CONCEPTO ---
if __name__ == "__main__":
    evaluador = EvaluadorMetricas()
    tweet_real = "The earthquake in Chile was terrifying, many buildings collapsed."
    tweet_sintetico_bueno = "A massive earthquake struck Chile today causing severe damage to structures."
    
    print("\n--- PRUEBA DE CALIDAD (SBERT) ---")
    calidad_buena = evaluador.calcular_calidad_sbert(tweet_sintetico_bueno, tweet_real)
    print(f"Calidad del tweet sintético: {calidad_buena:.4f}")