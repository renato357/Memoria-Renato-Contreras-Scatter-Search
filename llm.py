import ollama
import time

class MotorLLM:
    def __init__(self, model_name="llama3.1:8b"):
        """
        Inicializa el motor usando Ollama local.
        Ollama gestiona automáticamente la VRAM de la gráfica.
        """
        print(f"Iniciando Motor LLM con Ollama (Modelo: {model_name})...")
        self.model_name = model_name
        
    def invocar(self, prompt: str, system_prompt: str = "You are a prompt engineering assistant.", temp: float = 0.8) -> str:
        """
        Genera un texto controlando parámetros vitales para la reproducibilidad.
        """
        max_retries = 3
        for i in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    # Parámetros estrictos para la investigación
                    options={
                        'temperature': temp, # 0.8 para igualar el baseline de Sáez
                        'seed': 42,          # Fijamos la semilla para evitar aleatoriedad extrema
                        'num_predict': 150   # Límite de tokens de salida (ahorra cómputo)
                    }
                )
                return response['message']['content'].strip()
            except Exception as e:
                print(f"  [Error LLM] Intento {i+1}/{max_retries}: {e}")
                time.sleep(2)
        return ""

# --- PRUEBA RÁPIDA ---
if __name__ == "__main__":
    motor = MotorLLM()
    print("Enviando prompt de prueba...")
    resultado = motor.invocar("Write a 1-sentence tweet about the importance of synthetic data in AI.")
    print(f"\nRespuesta:\n{resultado}")