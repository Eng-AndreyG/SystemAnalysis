from sklearn.metrics import classification_report
import pandas as pd
import joblib
from pathlib import Path
from config import DATA_PROCESSED, MODELS_DIR
from modules.models import ToxicityModel
from preprocessing import load_and_preprocess
import sys

# Función principal del sistema de toxicidad
# Carga datos, entrena modelo y evalúa casos de prueba
def main():
    print("=== Sistema de Toxicidad ===")
    
    
    try:
        print("\n...Cargando datos...")
        train, test = load_and_preprocess()
        
        print("\nDistribución de clases:")
        print(f"- Total ejemplos: {len(train)}")
        print(f"- Tóxicos: {train['target_binary'].sum()} ({train['target_binary'].mean():.1%})")
        
        print("\nEntrenando modelo calibrado...")
        model = ToxicityModel()
        model.train(train['comment_text'], train['target_binary'])
        
        # Evaluación del modelo
        test_cases = [
            ("Neutro", "Hoy es un día soleado", 0),
            ("Neutro con emoción", "¡Qué hermoso día hace hoy!", 0),
            ("Insulto directo", "Eres un completo idiota", 1),
            ("Ataque identidad", "Los musulmanes son el problema", 1),
            ("Amenaza", "Voy a matarte si hablas", 1),
            ("Lenguaje obsceno", "Eres un maldito pedazo de mierda", 1),
            ("Falso positivo", "Orgullosamente musulmán", 0),
            ("Positivo con identidad", "Las mujeres son excelentes líderes", 0),
            ("Amenaza física", "Te voy a golpear", 1),
            ("Emoción positiva", "Me encanta este lugar", 0)
        ]
        
        print("\nPredicciones calculadas:")
        results = []
        # Iterar sobre los casos de prueba
        for name, text, expected in test_cases:
            proba = model.predict(text)
            prediction = 1 if proba >= 0.6 else 0
            correct = prediction == expected
            results.append((name, proba, prediction, expected, correct))
            
            color = "\033[92m" if correct else "\033[91m"
            reset = "\033[0m"
            print(f"{color}{'✅' if correct else '❌'} {name:<25}: {proba:.4f} ({'TÓXICO' if prediction else 'No tóxico'}){reset}")
        
        # Calcular promedio de precisión
        accuracy = sum(1 for r in results if r[4]) / len(results)
        print(f"\nPrecisión en casos críticos: {accuracy:.2%}")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / 'perfectly_calibrated_model.joblib'
        joblib.dump(model, model_path)
        print(f"\nModelo calibrado guardado en: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()