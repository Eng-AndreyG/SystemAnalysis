from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import re
from textblob import TextBlob

# Modelo de Toxicidad con sistema de reglas 
class ToxicityModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=12000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.7
            )),
            ('clf', LogisticRegression(
                penalty='l2',
                C=0.4,
                solver='liblinear',
                class_weight={0: 1, 1: 2},
                max_iter=2000
            ))
        ])
        
        # Sistema de reglas mejor calibrado
        self.rules = [
            # Reglas POSITIVAS 
            {
                'patterns': [
                    (r'¡?qué\s+(hermoso|bonito|lindo)\s+día!?', -0.8),
                    (r'me\s+encanta\s+esto', -0.7),
                    (r'estoy\s+feliz\s+por', -0.6)
                ],
                'type': 'positive'
            },
            
            # Reglas de AMENAZAS (alta prioridad)
            {
                'patterns': [
                    (r'\b(voy\s+a\s+matar|te\s+voy\s+a\s+|te\s+juro\s+que)\b', 0.8),
                    (r'\b(te\s+arrepentirás|vas\s+a\s+morir)\b', 0.75),
                    (r'\b(romper(é|e|emos|án)|golpear(é|e|emos|án))\b', 0.7)
                ],
                'type': 'threat'
            },
            
            # Reglas de IDENTIDAD POSITIVA
            {
                'patterns': [
                    (r'orgullosamente\s+(musulmán|cristiano|gay)', -0.9),
                    (r'(mujeres|hombres)\s+son\s+excelentes', -0.8)
                ],
                'type': 'positive_identity'
            },
            
            # Otras reglas tóxicas
            {
                'patterns': [
                    (r'\b(idiota|estúpido|imbécil)\b', 0.4),
                    (r'\b(puta|mierda|diablo)\b', 0.35),
                    (r'\b(musulmanes|gays)\s+son\s+el\s+problema\b', 0.6)
                ],
                'type': 'toxic'
            }
        ]

    def train(self, X, y):
        """Entrena el modelo con ajuste de parámetros"""
        self.pipeline.fit(X, y)
    
    def _apply_rules(self, text):
        """Aplica el sistema de reglas en orden prioritario"""
        text_lower = text.lower()
        adjustments = []
        
        for rule_group in self.rules:
            max_boost = 0
            for pattern, boost in rule_group['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if rule_group['type'] in ['threat', 'toxic']:
                        if boost > max_boost:
                            max_boost = boost
                    else:  # Reglas positivas
                        if boost < max_boost:  # Buscamos la reducción más fuerte
                            max_boost = boost
            
            if max_boost != 0:
                adjustments.append((rule_group['type'], max_boost))
                # Si encontramos una amenaza o contenido positivo fuerte, paramos aquí
                if rule_group['type'] in ['threat', 'positive', 'positive_identity'] and abs(max_boost) > 0.7:
                    break
        
        return adjustments
    
    def predict(self, X):
        """Predicción con sistema de reglas calibrado"""
        if isinstance(X, str):
            X = [X]
        
        probas = self.pipeline.predict_proba(X)[:, 1]
        
        for i, text in enumerate(X if isinstance(X, list) else [X]):
            base_proba = probas[i]
            adjustments = self._apply_rules(text)
            
            # Aplicar ajustes en orden
            for rule_type, boost in adjustments:
                if rule_type in ['threat', 'toxic']:
                    base_proba = min(base_proba + boost, 1.0)
                else:  # Reglas positivas
                    base_proba = max(base_proba + boost, 0.0)
                
                # Si es una regla determinante, no aplicar más ajustes
                if abs(boost) > 0.7:
                    break
            
            probas[i] = base_proba
        
        return probas if len(probas) > 1 else probas[0]