import ollama
import time

class MotorLLM:
    def __init__(self, model_name="llama3.1:8b", semilla=None, max_tokens_salida=150):
        """
        Inicializa el motor usando Ollama local.
        Ahora todos los parámetros duros se pueden controlar desde el main.
        """
        print(f"Iniciando Motor LLM con Ollama (Modelo: {model_name})...")
        self.model_name = model_name
        self.semilla = semilla
        self.max_tokens_salida = max_tokens_salida
        
    def invocar(self, prompt: str, system_prompt: str = "You are a prompt engineering assistant.", temp: float = 0.8) -> str:
        """
        Genera un texto controlando parámetros vitales para la reproducibilidad.
        La temperatura (temp) se recibe por llamada, ya que varía según la fase (Exploración vs Corrección).
        """
        # Configuramos las opciones base
        opciones_llm = {
            'temperature': temp,
            'num_predict': self.max_tokens_salida
        }
        
        # Si la función principal manda una semilla, forzamos el determinismo en Llama
        if self.semilla is not None:
            opciones_llm['seed'] = self.semilla

        max_retries = 3
        for i in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    options=opciones_llm
                )
                return response['message']['content'].strip()
            except Exception as e:
                print(f"  [Error LLM] Intento {i+1}/{max_retries}: {e}")
                time.sleep(2)
        return ""

# --- PRUEBA RÁPIDA ---
if __name__ == "__main__":
    motor = MotorLLM(semilla=42)
    print("Enviando prompt de prueba...")
    resultado = motor.invocar("Write a 1-sentence tweet about the importance of synthetic data in AI.")
    print(f"\nRespuesta:\n{resultado}")