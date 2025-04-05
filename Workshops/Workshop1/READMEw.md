
# Workshop 1

## Kaggle Systems Engineering Analysis of Jigsaw Unintended Bias in Toxicity Classification

## Why this competition?

This competition was chosen primarily for its dataset, which is mostly natural language processing (NLP) and numerical data, which are easier to analyze to identify system relationships and how the dataset is structured. Furthermore, since it is an artificial intelligence model, this type of language is easier to analyze than the use of images commonly seen in datasets from other competitions that were of interest to us. Another reason we chose this competition is because each comment in the dataset, when classified, makes it much easier for us to understand the topic on which the competition is based. This is unlike other competitions, such as identifying whales or cars, that didn't allow us to identify them in the dataset. This saves us the huge step of having to thoroughly investigate the topic to perform the systematic analysis.
Do you think so? Do you think you can improve anything?

---

## Competition Description

This Kaggle competition aims to build a model capable of identifying toxic online comments while minimizing bias toward identity groups such as gender, race, or religion. The target variable represents the fraction of annotators who considered a comment to be toxic.

In addition to standard prediction, the model's performance in terms of fairness is evaluated using metrics such as ROC-AUC per subgroup and intersectional bias metrics.

---

## Dataset Components

- **`train.csv`**: Contains the training data with comments, toxicity labels and identity mentions, each characterized with a score.
- **`test.csv`**: Unlabeled file used for the final prediction.
- **`identity_annotations.csv`, `toxicity_annotations.csv`**: Individual annotator data.

### Features:

- `comment_text`: The text of the comment.
- Toxicity subtypes: `severe_toxicity`, `obscene`, `insult`, `threat`, etc.
- Identities: `male`, `female`, `transgender`, `black`, `white`, `Christian`, `Muslim`, etc.

### Target Variable:

- `target`: Fraction of annotators who marked the comment as toxic (continuous value from 0 to 1).

---

## Systems Analysis Methodology

This work focused on conducting a systemic analysis of the competition, considering its key components and how they interact within a complex system. The methodology was developed in the following steps:

1. **Identification of Key Elements**
The dataset files were analyzed to understand their structure, content, data type, and relationships between variables. Input variables, expected outputs, and labels related to toxicity and identity mentions were identified.

2. **Mapping System Relationships**
A conceptual model of the data flow was constructed, from input comments to toxicity prediction, including the interaction between identity mentions and evaluator annotations. The system constraints imposed by the competition rules were also analyzed.

3. **Application of Systems Engineering Principles**
System requirements were defined from an analytical perspective, including criteria such as analytical accuracy, bias minimization, and data flow structure. Aspects of the system's lifecycle in a realistic environment were also considered.

4. **Sensitivity Analysis**
We evaluated how changes in input parameters or dataset configurations could alter system relationships and outputs. Variables such as class balance and the presence or absence of identities were considered.

5. **Exploration from Chaos and Complexity Theory**
Chaotic and unpredictable elements in the data were discussed, such as human subjectivity in annotations, the use of sarcasm in language, and the influence of identities on the perception of toxicity.

---

## Credits

Work carried out as part of a systemic analysis of machine learning competition on Kaggle.
Authors:

Laura Nathaly Paez Cifuentes
Andrey Camilo Gonzales Caceres
Hugo Mojica Angarita

---
