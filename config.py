
#BASE_MODEL = "roberta-base"
# BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"
# BASE_MODEL = "allenai/scibert_scivocab_uncased"
#BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
BASE_MODEL = "allenai/biomed_roberta_base"

USE_CRF = False

# Paths & dataset
DATASET_NAME          = "tner/bc5cdr"
YOUR_MODEL_PATH       = "./final_clinical_ner_crf_model"
PRETRAINED_BASELINE   = "tner/roberta-large-bc5cdr"

# Evaluation settings
DATASET_PERCENTAGE       = 0.5
RANDOM_DATASET_VIZ_COUNT = 5

# Labels (BIO schema — matches T-NER / BC5CDR)
LABEL_LIST = ["O", "B-CHEMICAL", "B-DISEASE", "I-DISEASE", "I-CHEMICAL"]
ID2LABEL   = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID   = {label: i for i, label in enumerate(LABEL_LIST)}

# Visualisation colours
COLORS = {
    "CHEMICAL": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "DISEASE":  "linear-gradient(90deg, #ff9a8d, #ff6961)",
    "DOSAGE":   "linear-gradient(90deg, #feca57, #ff9ff3)",
}

# Custom inference sentences
CUSTOM_SENTENCES = [
    "The patient was prescribed 50mg of Aspirin for the headache.",
    "Significant side effects were noted after administering 10ml of Doxorubicin.",
    "History of myocardial infarction and hypertension.",
    "Injection of 0.5ml epinephrine resolved the anaphylaxis immediately.",
]