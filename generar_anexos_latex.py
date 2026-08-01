import json
import os
import numpy as np

def escape_latex(text):
    if not isinstance(text, str):
        return str(text)
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('^', '\\textasciicircum{}')
    return text

def generar_anexos_f_g():
    ruta_exp4 = "exp4_temperaturas.json"
    ruta_exp5 = "exp5_inicializacion.json"
    ruta_exp1 = os.path.join("Resultados - Exp 1", "exp1_base.json")
    
    ruta_out = r"C:\Users\renac\Downloads\Memoria\Memoria_Renato_Scatter\tesis-udp\ejemplo\anexos.tex"
    
    latex = []
    
    # --- ANEXO F: Exp 4 ---
    latex.append(r"\newpage")
    latex.append(r"\section{Anexo F: Impacto de la Temperatura en Operadores de Mejora Local (Experimento 4)}")
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{c | c c c | c}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Temp ($T$)} & \textbf{Corrida 1} & \textbf{Corrida 2} & \textbf{Corrida 3} & \textbf{Promedio Final SBERT} \\")
    latex.append(r"\midrule")
    
    if os.path.exists(ruta_exp4) and os.path.exists(ruta_exp1):
        with open(ruta_exp4, "r", encoding="utf-8") as f:
            d4 = json.load(f)
        with open(ruta_exp1, "r", encoding="utf-8") as f:
            runs_base = json.load(f)
            
        sberts_base = [r['historial_convergencia'][-1]['mejor_sbert'] for r in runs_base]
        
        datos_temp = {}
        for t_key in d4.keys():
            temp_val = float(t_key.replace("conf_T", ""))
            runs = d4[t_key]
            datos_temp[temp_val] = [r['historial_convergencia'][-1]['mejor_sbert'] for r in runs]
            
        datos_temp[0.2] = sberts_base[:3] # Tomamos las primeras 3 para igualar columnas
        
        for t in sorted(datos_temp.keys()):
            runs = datos_temp[t]
            prom = np.mean(runs)
            if t == 0.0:
                # Negrita para el ganador
                latex.append(f"{t} & \\textbf{{{runs[0]:.4f}}} & \\textbf{{{runs[1]:.4f}}} & \\textbf{{{runs[2]:.4f}}} & \\textbf{{{prom:.4f}}} \\\\")
            else:
                latex.append(f"{t} & {runs[0]:.4f} & {runs[1]:.4f} & {runs[2]:.4f} & {prom:.4f} \\\\")
    else:
        latex.append(r"\multicolumn{5}{c}{Datos del Exp 4 no encontrados} \\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Métricas SBERT finales obtenidas por Scatter Search al variar la temperatura del operador de parafraseo (GrIPS).}")
    latex.append(r"\label{tab:anexo_f}")
    latex.append(r"\end{table}")
    
    # --- ANEXO G: Exp 5 ---
    latex.append(r"\newpage")
    latex.append(r"\section{Anexo G: Impacto de la Temperatura en la Generación Inicial (Experimento 5)}")
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{c | c c | c}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Temp ($T$)} & \textbf{SBERT Máximo (Promedio)} & \textbf{SBERT Promedio (Población)} & \textbf{Diversidad Léxica (Promedio)} \\")
    latex.append(r"\midrule")
    
    if os.path.exists(ruta_exp5):
        with open(ruta_exp5, "r", encoding="utf-8") as f:
            d5 = json.load(f)
            
        for t_key in sorted(d5.keys()):
            temp_val = float(t_key.replace("conf_T", ""))
            runs = d5[t_key]
            s_max = np.mean([r['sbert_maximo'] for r in runs])
            s_avg = np.mean([r['sbert_promedio'] for r in runs])
            div = np.mean([r['diversidad_lexica'] for r in runs])
            
            if temp_val == 0.9:
                latex.append(f"{temp_val} & \\textbf{{{s_max:.4f}}} & \\textbf{{{s_avg:.4f}}} & \\textbf{{{div:.1f}}} \\\\")
            else:
                latex.append(f"{temp_val} & {s_max:.4f} & {s_avg:.4f} & {div:.1f} \\\\")
    else:
        latex.append(r"\multicolumn{4}{c}{Datos del Exp 5 no encontrados} \\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Efecto de la temperatura en la población inicial ($P=40$). Se observa un aumento sostenido en la Diversidad Léxica a mayores temperaturas.}")
    latex.append(r"\label{tab:anexo_g}")
    latex.append(r"\end{table}")
    
    # Escribir el archivo
    with open(ruta_out, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))
    print(f"Anexos F y G generados con exito en: {ruta_out}")

if __name__ == "__main__":
    generar_anexos_f_g()
