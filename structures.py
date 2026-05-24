from dataclasses import dataclass, field
from typing import List

@dataclass
class PromptSolution:
    """
    Representa un individuo dentro de nuestra población o RefSet.
    Maneja de forma estructurada los módulos de Rol y Tarea propuestos en la arquitectura.
    """
    rol: str
    tarea: str
    prompt_completo: str = ""
    dato_generado: str = ""      # El tweet o dato sintético que escupe Llama 3
    
    # Métricas Multiobjetivo
    score_sbert: float = 0.0     # Calidad semántica
    score_bleu: float = 0.0      # Distancia léxica
    
    # Trazabilidad
    origen: str = "inicializacion" # Ej: "cross_swap", "grips_delete", etc.
    generacion: int = 0
    
    # Para el operador 'Add' de GrIPS
    frases_borradas: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Si instanciamos pasando rol y tarea pero no el prompt completo, lo armamos.
        if not self.prompt_completo and self.rol and self.tarea:
            self.prompt_completo = f"Role: {self.rol}. Task: {self.tarea}"

    @property
    def texto_para_cruce(self) -> str:
        """Retorna solo la instrucción para los operadores semánticos"""
        return self.tarea