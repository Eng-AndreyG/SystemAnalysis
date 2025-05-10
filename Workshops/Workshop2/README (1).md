
# Workshop 2 – Systems Design for Toxic Comment Classification

This repository contains the development process and system design for Workshop 2 of the *Systems Analysis & Design* course (Semester 2025-I). The workshop focuses on designing a robust system to participate in the Kaggle competition **"Jigsaw Unintended Bias in Toxicity Classification"**, based on the system analysis carried out in Workshop 1.

## Objectives

- Translate the findings from the previous analysis into a concrete, modular, and scalable system design.
- Incorporate strategies to address sensitivity and chaos caused by subjective annotations.
- Propose an implementation sketch using modern tools and engineering principles.

## Key Insights from Workshop 1

- **Constraints**: Resource limitations, single dataset availability, and ethical concerns such as subgroup bias.
- **Chaos & Sensitivity**: Arise from human subjectivity, ambiguous language, and small variations in identity-related inputs.
- **Data Nature**: Multidimensional data with 7 different toxicity categories, subject to annotator inconsistencies.

## System Architecture

The design is composed of modular components, each with a specific responsibility. Key modules include:

- **Data ingestion and validation**: `Labels`, `DataValidator`, `ColumnSelector`
- **Text preprocessing**: `TextCleaner`, `Lemmatizer`
- **Feature extraction**: `TFIDVectorizer`, `NumericFeatureMerger`, `IdentityFlagExtractor`
- **Modeling**: `ToxicityPredictor`, `PredictorCollector`, `PredictionAggregator`
- **Bias and chaos management**: `IdentityAttackChecker`, `AnnotatorWeightCalculator`
- **Output formatting**: `CVSFormatter`

All components interact through clean, decoupled interfaces (mainly Pandas DataFrames), allowing scalability and easy maintenance.

![System Architecture](./img/architecture.png)

## Technical Stack

- **Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn

This stack supports structured data manipulation, lightweight modeling, and sensitive subgroup analysis under the competition constraints (e.g., RAM limits and fairness metrics).

## Chaos & Sensitivity Mitigation

To handle unpredictable behavior caused by annotation variability and implicit biases:

- `IdentityAttackChecker`: Detects subtle harmful patterns not explicitly labeled.
- `AnnotatorWeightCalculator`: Adjusts predictions by assigning dynamic weights to annotator decisions.
- Non-linear feedback risks and bifurcations are mitigated by reinforcing fair metric evaluation and subgroup-aware performance validation.

## Documentation

The full system design is available in the `Workshop2_Design` PDF:

📎 [System Design Document (PDF)](./Workshop2_Design/Workshop2_Design.pdf)  

## Deliverables Summary

- [x] System Design Document (PDF)
- [x] Architecture Diagrams
- [x] README.md with full development description

## Credits

Work carried out as part of a systemic analysis of machine learning competition on Kaggle.
Authors:

Laura Nathaly Paez Cifuentes, 
Andrey Camilo Gonzales Caceres, 
Hugo Mojica Angarita
