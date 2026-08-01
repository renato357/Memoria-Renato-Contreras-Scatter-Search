import json
import os
import glob

files = [
    r"Resultados - Exp 1\exp1_base.json",
    r"Resultados - Exp 2\exp2_sensibilidad.json",
    r"Resultados - Exp 3\exp3_cruzada_off.json",
    r"exp4_temperaturas.json",
    r"exp5_inicializacion.json"
]

def extract_tweets(data):
    tweets = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ['refset_final', 'poblacion_inicial', 'poblacion', 'historial_convergencia']:
                tweets.extend(extract_tweets(v))
            elif isinstance(v, (dict, list)):
                tweets.extend(extract_tweets(v))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'tweet_generado' in item and 'tweet_real_match' in item:
                tweets.append(item)
            elif isinstance(item, (dict, list)):
                tweets.extend(extract_tweets(item))
    return tweets

output = []
for f_path in files:
    full_path = os.path.join(r"c:\Users\renac\Downloads\Memoria\Memoria-Renato-Contreras-Scatter-Search", f_path)
    if not os.path.exists(full_path):
        continue
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    candidates = extract_tweets(data)
    
    good_ones = []
    for c in candidates:
        gen = c.get('tweet_generado', '')
        sbert = c.get('sbert', 0)
        
        if sbert > 0.75 and '\n' not in gen and len(gen) < 280:
            if not gen.startswith('I can') and not gen.startswith('Here'):
                if '**' not in gen and 'I\'m sorry' not in gen:
                    good_ones.append(c)
    
    good_ones.sort(key=lambda x: x.get('sbert', 0), reverse=True)
    
    # Check for slightly lower threshold if none found
    if not good_ones:
        for c in candidates:
            gen = c.get('tweet_generado', '')
            sbert = c.get('sbert', 0)
            if sbert > 0.65 and '\n' not in gen and len(gen) < 280:
                if not gen.startswith('I can') and not gen.startswith('Here') and '**' not in gen:
                    good_ones.append(c)
        good_ones.sort(key=lambda x: x.get('sbert', 0), reverse=True)

    if good_ones:
        output.append(f"--- File: {f_path} ---")
        for g in good_ones[:5]:
            output.append(f"SBERT: {g.get('sbert')}")
            output.append(f"Gen: {g.get('tweet_generado')}")
            output.append(f"Real: {g.get('tweet_real_match')}")
            output.append("-" * 20)

with open(r"c:\Users\renac\Downloads\Memoria\Memoria-Renato-Contreras-Scatter-Search\scratch\out_utf8.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
