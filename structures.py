from dataclasses import dataclass, field
from typing import List

@dataclass
class PromptSolution:
    """
    Representa un individuo dentro de nuestra población o RefSet.
    Maneja de forma estructurada los módulos de Rol y Tarea.
    """
    rol: str
    tarea: str
    prompt_completo: str = ""
    dato_generado: str = ""      
    texto_referencia_match: str = ""  # El tweet real que más se pareció
    
    score_sbert: float = 0.0     # Calidad semántica
    
    # Trazabilidad
    origen: str = "inicializacion" 
    generacion: int = 0
    
    # Para el operador 'Add' de GrIPS
    frases_borradas: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.prompt_completo and self.rol and self.tarea:
            self.prompt_completo = f"Role: {self.rol}. Task: {self.tarea}"

    @property
    def texto_para_cruce(self) -> str:
        """Retorna solo la instrucción para los operadores semánticos"""
        return self.tarea