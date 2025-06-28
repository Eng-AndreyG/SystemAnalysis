# Workshop 3 - Toxic Comment Classification Simulation

## What This Does
This is a complete simulation of our toxicity detection system for Kaggle comments. It tests:
- Basic toxicity detection
- Handling of identity-related content
- Sarcasm/context understanding
- System stability

## Key Components
1. **Text Cleaner** - Prepares raw text while keeping important signals
2. **Identity Detector** - Flags mentions of sensitive groups (gender, race, etc.)
3. **Context Analyzer** - Catches sarcasm and sentiment
4. **Toxicity Model** - Combines machine learning with rule-based adjustments

## How It Works
The system processes each comment through four steps:
Raw Text → Clean → Check Identities → Analyze Context → Final Classification


## What We Tested
We ran three types of tests:

### 1. Basic Classification
| Test Case          | Result |
|--------------------|--------|
| Direct insult      | ✅ 98% accurate |
| Neutral comment    | ✅ 95% accurate |
| Threat             | ✅ 92% accurate |

### 2. Identity Handling
| Identity Group | False Alarm Rate |
|----------------|------------------|
| Gender         | 12%              |
| Religion       | 15%              |
| Race           | 18%              |

### 3. Tricky Cases
| Case Type       | Success Rate |
|-----------------|--------------|
| Sarcasm        | 75%          |
| Positive ID    | 88%          |
| Mixed Context  | 65%          |

## How to Run
1. Make sure you have Python 3.8+
2. Install requirements:
```bash
pip install textblob scikit-learn pandas

## What We Found

**What worked well:**
- Detects obvious toxicity with 94% accuracy  
- Reduces bias compared to baseline  
- Handles simple threats effectively  

**Problems we found:**
- Struggles with cultural references  
- Misses some sarcasm (only 75% caught)  
- Inconsistent with mixed tone comments  

## Files Included
- `toxicity_simulation.py` - Main simulation code  
- `workshop3_report.pdf` - Full technical report 

## Team
- [Andrey Camilo Gonzlez Caceres]  
- [Hugo Mojica Angarita]  
- [Laura Paez Cifuentes]  

Universidad Distrital Francisco José de Caldas  
Systems Analysis & Design - 2025


