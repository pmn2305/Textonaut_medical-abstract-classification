# Medical Abstract Section Classification using BERT

This project demonstrates transfer learning and fine-tuning of a BERT-based model for a multi-class text classification task.
The task is to classify sentences from medical research abstracts into their corresponding rhetorical sections such as:

BACKGROUND

OBJECTIVE

METHODS

RESULTS

CONCLUSIONS

The model is fine-tuned using PyTorch (custom training loop) on a subset of the PubMed RCT dataset.

📊 Dataset

Dataset used:
pietrolesci/pubmed-200k-rct (from Hugging Face)

Each sample consists of:

text: a sentence from a medical abstract

labels: the section label for the sentence

A subset of the dataset was used for training and evaluation due to computational constraints:

Training samples: 5,000

Test samples: 1,000

Exploratory Data Analysis (EDA)

The class distribution of the training set was visualized using a bar chart.
The dataset shows moderate class imbalance, with METHODS and RESULTS having higher frequencies compared to OBJECTIVE and CONCLUSIONS.

Therefore, weighted evaluation metrics (Precision, Recall, F1-score) were used for performance reporting.

Model Architecture

Pretrained model: bert-base-uncased

Classification head: Fully connected linear layer on top of BERT’s pooled output

Dropout applied for regularization

Architecture:

Text → BERT → [CLS] pooled output → Linear layer → Class probabilities

Loss function: CrossEntropyLoss
Optimizer: AdamW
Max sequence length: 128 tokens

Training

The model was trained using a custom PyTorch training loop (no Hugging Face Trainer API was used).

Training details:

Epochs: 2

Batch size: 8

Learning rate: 2e-5

Tokenizer: BertTokenizer

📈 Evaluation Metrics

The model was evaluated on the test set using:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Confusion Matrix

Results:

Accuracy: 0.844

Precision (weighted): 0.847

Recall (weighted): 0.844

F1-score (weighted): 0.842

The confusion matrix shows strong performance on METHODS and RESULTS classes.
Most misclassifications occur between semantically similar categories such as:

BACKGROUND vs OBJECTIVE

RESULTS vs CONCLUSIONS

Inference Pipeline

A function predict_text(text: str) was implemented to perform inference on raw input sentences.
It returns:

Predicted class label

Confidence score (softmax probability)

The model was tested using manually written examples that do not explicitly contain section keywords, demonstrating that it learned semantic patterns rather than relying only on surface-level cues.

▶️ How to Run

Install dependencies:

pip install datasets transformers scikit-learn matplotlib seaborn tqdm torch


Open the notebook:

jupyter notebook


Run all cells in order to:

Load the dataset

Perform EDA

Train the model

Evaluate performance

Test inference

📁 Repository Structure
.
├── notebook.ipynb
└── README.md


All code, including data loading, training, evaluation, and inference, is contained in the notebook.

📌 Notes

Approach:

Fine-tuned a pre-trained BERT-based model (bert-base-uncased) on the PubMed 200k RCT dataset for multi-class medical abstract sentence classification.
Each sentence was tokenized using the BERT tokenizer and mapped to one of five labels: BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS.

Training was done using PyTorch only, without using Hugging Face Trainer APIs, to explicitly demonstrate the fine-tuning process including:

Forward pass

Loss computation

Backpropagation

Optimization using AdamW

Evaluated the model using Accuracy, Precision, Recall, F1-score (weighted), and a Confusion Matrix.

Assumptions:

The dataset sentences are independent and can be classified without requiring surrounding context.

The provided labels accurately represent the rhetorical role of each sentence in a medical abstract.

A reduced subset of the dataset was sufficient to demonstrate effective fine-tuning within reasonable training time and hardware constraints.

Pre-trained BERT embeddings are suitable for biomedical text classification without domain-specific pretraining (e.g., BioBERT).

Observations:

The model achieved strong performance despite limited fine-tuning, showing that transfer learning is effective for structured medical text.

Sentences with explicit keywords (e.g., “patients”, “results”, “conclusion”) were easier for the model to classify.

Some ambiguity was observed between BACKGROUND and OBJECTIVE classes, likely due to overlapping language patterns.

Confidence scores were generally higher for clearly structured scientific statements and lower for shorter or more generic sentences.

Conclusion

This project shows that a pretrained BERT model can be successfully fine-tuned to classify medical abstract sections with strong performance.
The approach can be generalized to other text classification tasks by changing only the dataset and label set.
