import json
import os
import numpy as np

def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(set1.intersection(set2)) / len(union)

def escape_latex(text):
    chars = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
    }
    for k, v in chars.items():
        text = text.replace(k, v)
    return text

def generar_anexos():
    ruta_exp1 = os.path.join("Resultados - Exp 1", "exp1_base.json")
    ruta_exp2 = os.path.join("Resultados - Exp 2", "exp2_sensibilidad.json")
    ruta_exp3 = os.path.join("Resultados - Exp 3", "exp3_cruzada_off.json")
    ruta_out = r"C:\Users\renac\Downloads\Memoria\Memoria_Renato_Scatter\tesis-udp\ejemplo\anexos.tex"
    
    latex = []
    latex.append(r"\appendix")
    latex.append(r"\chapter{Anexos de Resultados Experimentales}")
    latex.append(r"\label{ch:anexos}")
    latex.append("")
    latex.append(r"Este apartado documenta los resultados detallados que respaldan el análisis empírico presentado en el Capítulo \ref{ch:implementacion_evaluacion}.")
    latex.append("")

    # ==========================================
    # ANEXO A: HISTORIAL DE CONVERGENCIA (EXP 1)
    # ==========================================
    latex.append(r"\section{Anexo A: Historial de Convergencia (Experimento 1)}")
    if os.path.exists(ruta_exp1):
        with open(ruta_exp1, "r", encoding="utf-8") as f:
            runs1 = json.load(f)
        
        latex.append(r"\begin{table}[H]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{cccccc}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Generación} & \textbf{Corrida 1} & \textbf{Corrida 2} & \textbf{Corrida 3} & \textbf{Corrida 4} & \textbf{Corrida 5} \\")
        latex.append(r"\midrule")
        
        # Asumimos que todas tienen 10 generaciones (0 a 10)
        num_gens = len(runs1[0]['historial_convergencia'])
        for g in range(num_gens):
            fila = [str(g)]
            for run in runs1:
                hist = run['historial_convergencia']
                mejor = hist[g]['mejor_sbert'] if g < len(hist) else hist[-1]['mejor_sbert']
                fila.append(f"{mejor:.4f}")
            latex.append(" & ".join(fila) + r" \\")
            
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\caption{Evolución del mejor puntaje SBERT por generación a lo largo de 5 ejecuciones independientes (Configuración Base).}")
        latex.append(r"\label{tab:anexo_a}")
        latex.append(r"\end{table}")
        
        latex.append("")
        latex.append(r"\noindent \textbf{Tiempos de ejecución:} ")
        tiempos = [f"C{i+1}: {r['configuracion']['tiempo_ejecucion_minutos']:.2f} min" for i, r in enumerate(runs1)]
        latex.append(", ".join(tiempos) + ".")
        latex.append("")

    # ==========================================
    # ANEXO B: EJEMPLOS CUALITATIVOS (EXP 1)
    # ==========================================
    latex.append(r"\newpage")
    latex.append(r"\section{Anexo B: Ejemplos Cualitativos de Instrucciones (Experimento 1)}")
    
    if os.path.exists(ruta_exp1):
        # Buscamos la mejor corrida (mayor SBERT en última generación)
        mejor_run = max(runs1, key=lambda x: x['historial_convergencia'][-1]['mejor_sbert'])
        # Top 5 individuos del refset final
        top5 = sorted(mejor_run['refset_final'], key=lambda x: x.get('sbert', 0.0), reverse=True)[:5]
        
        latex.append(r"A continuación se exponen las cinco mejores instrucciones generadas por la metaheurística SS-GrIPS durante la corrida más exitosa del Experimento Base, contrastadas directamente con el texto objetivo.")
        latex.append("")
        
        for i, ind in enumerate(top5):
            gen = ind.get('tweet_generado', '')
            real = ind.get('tweet_real_match', '')
            sbert = ind.get('sbert', 0.0)
            jacc = jaccard_similarity(gen, real)
            
            latex.append(r"\begin{table}[H]")
            latex.append(r"\centering")
            latex.append(r"\begin{tabular}{p{0.45\textwidth} p{0.45\textwidth}}")
            latex.append(r"\toprule")
            latex.append(r"\textbf{Texto Objetivo (Original)} & \textbf{Instrucción Generada (SS-GrIPS)} \\")
            latex.append(r"\midrule")
            latex.append(f"{escape_latex(real)} & {escape_latex(gen)} \\\\")
            latex.append(r"\midrule")
            latex.append(r"\multicolumn{2}{c}{\textbf{Métricas:} SBERT = " + f"{sbert:.4f}" + r" | Jaccard = " + f"{jacc*100:.2f}\\%" + r"} \\")
            latex.append(r"\bottomrule")
            latex.append(r"\end{tabular}")
            latex.append(f"\\caption{{Par cualitativo número {i+1} extraído del Top 5 de la mejor población.}}")
            latex.append(r"\end{table}")
            latex.append("")

    # ==========================================
    # ANEXO C: SENSIBILIDAD (EXP 2)
    # ==========================================
    latex.append(r"\newpage")
    latex.append(r"\section{Anexo C: Datos de Sensibilidad (Experimento 2)}")
    if os.path.exists(ruta_exp2):
        with open(ruta_exp2, "r", encoding="utf-8") as f:
            data2 = json.load(f)
            
        orden = ['conf_G5_R10', 'conf_G10_R5', 'conf_G10_R10', 'conf_G15_R10', 'conf_G10_R15']
        
        latex.append(r"\begin{table}[H]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{l c c c | c c}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Configuración} & \textbf{Corrida 1} & \textbf{Corrida 2} & \textbf{Corrida 3} & \textbf{Prom. SBERT} & \textbf{Prom. Tiempo} \\")
        latex.append(r"\midrule")
        
        for c in orden:
            if c in data2:
                runs = data2[c]
                sberts = [r['historial_convergencia'][-1]['mejor_sbert'] for r in runs]
                tiempos = [r['configuracion']['tiempo_ejecucion_minutos'] for r in runs]
                
                # Completar si hay menos de 3 corridas
                while len(sberts) < 3:
                    sberts.append(0.0)
                
                prom_sbert = np.mean(sberts)
                prom_tiempo = np.mean(tiempos)
                
                fila = f"{escape_latex(c)} & {sberts[0]:.4f} & {sberts[1]:.4f} & {sberts[2]:.4f} & \textbf{{{prom_sbert:.4f}}} & \textbf{{{prom_tiempo:.1f} m}} \\\\"
                latex.append(fila)
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\caption{Métricas detalladas para el análisis de sensibilidad según configuración de hiperparámetros.}")
        latex.append(r"\label{tab:anexo_c}")
        latex.append(r"\end{table}")
        latex.append("")

    # ==========================================
    # ANEXO D: DATA LEAKAGE (EXP 3)
    # ==========================================
    latex.append(r"\newpage")
    latex.append(r"\section{Anexo D: Efecto Data Leakage (Experimento 3)}")
    if os.path.exists(ruta_exp1) and os.path.exists(ruta_exp3):
        with open(ruta_exp3, "r", encoding="utf-8") as f:
            runs3 = json.load(f)
            
        # Tomamos 3 corridas del exp1 (ON) y 3 del exp3 (OFF)
        runs_on = runs1[:3]
        runs_off = runs3[:3]
        
        latex.append(r"\begin{table}[H]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{c | c c | c c}")
        latex.append(r"\toprule")
        latex.append(r" & \multicolumn{2}{c|}{\textbf{Validación Cruzada (ON)}} & \multicolumn{2}{c}{\textbf{Data Leakage (OFF)}} \\")
        latex.append(r"\textbf{Corrida} & \textbf{SBERT} & \textbf{Jaccard} & \textbf{SBERT} & \textbf{Jaccard} \\")
        latex.append(r"\midrule")
        
        sum_sbert_on = sum_jacc_on = sum_sbert_off = sum_jacc_off = 0
        
        for i in range(3):
            # ON
            sbert_on = runs_on[i]['historial_convergencia'][-1]['mejor_sbert']
            jaccs_on = [jaccard_similarity(ind.get('tweet_generado',''), ind.get('tweet_real_match','')) for ind in runs_on[i]['refset_final']]
            jacc_on = np.mean(jaccs_on)
            
            # OFF
            sbert_off = runs_off[i]['historial_convergencia'][-1]['mejor_sbert']
            jaccs_off = [jaccard_similarity(ind.get('tweet_generado',''), ind.get('tweet_real_match','')) for ind in runs_off[i]['refset_final']]
            jacc_off = np.mean(jaccs_off)
            
            sum_sbert_on += sbert_on; sum_jacc_on += jacc_on
            sum_sbert_off += sbert_off; sum_jacc_off += jacc_off
            
            latex.append(f"{i+1} & {sbert_on:.4f} & {jacc_on*100:.2f}\\% & {sbert_off:.4f} & {jacc_off*100:.2f}\\% \\\\")
            
        latex.append(r"\midrule")
        prom_sbert_on = sum_sbert_on/3; prom_jacc_on = sum_jacc_on/3
        prom_sbert_off = sum_sbert_off/3; prom_jacc_off = sum_jacc_off/3
        latex.append(f"\\textbf{{Promedio}} & \\textbf{{{prom_sbert_on:.4f}}} & \\textbf{{{prom_jacc_on*100:.2f}\\%}} & \\textbf{{{prom_sbert_off:.4f}}} & \\textbf{{{prom_jacc_off*100:.2f}\\%}} \\\\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\caption{Comparativa directa del influjo de métricas al evaluar el algoritmo sobre su propio conjunto de datos (Data Leakage).}")
        latex.append(r"\label{tab:anexo_d}")
        latex.append(r"\end{table}")
        latex.append("")

    # ==========================================
    # ANEXO E: ESTADÍSTICAS GRIPS (EXP 1)
    # ==========================================
    latex.append(r"\newpage")
    latex.append(r"\section{Anexo E: Rendimiento de los Operadores GrIPS (Experimento 1)}")
    if os.path.exists(ruta_exp1):
        latex.append(r"\begin{table}[H]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{l c c c c}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Operador} & \textbf{Intentos Totales} & \textbf{Éxitos Totales} & \textbf{Tasa de Éxito} & \textbf{Aporte SBERT Total} \\")
        latex.append(r"\midrule")
        
        totales = {"grips_delete": [0,0,0.0], "grips_swap": [0,0,0.0], "grips_paraphrase": [0,0,0.0]}
        for run in runs1:
            for op in totales.keys():
                st = run["stats_grips"].get(op, {})
                totales[op][0] += st.get("intentos", 0)
                totales[op][1] += st.get("exitos", 0)
                totales[op][2] += st.get("mejora_acumulada", 0.0)
                
        for op, vals in totales.items():
            intentos = vals[0]
            exitos = vals[1]
            mejora = vals[2]
            tasa = (exitos / intentos * 100) if intentos > 0 else 0
            latex.append(f"{escape_latex(op)} & {intentos} & {exitos} & {tasa:.2f}\\% & {mejora:.4f} \\\\")
            
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\caption{Estadísticas acumuladas de los operadores de edición tras 5 corridas. El operador Add fue retirado por ineficacia empírica.}")
        latex.append(r"\label{tab:anexo_e}")
        latex.append(r"\end{table}")

    # Escribir el archivo
    with open(ruta_out, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))
    print(f"Anexos generados con exito en: {ruta_out}")

if __name__ == "__main__":
    generar_anexos()
