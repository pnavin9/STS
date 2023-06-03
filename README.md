Apologies for the oversight. Here's an updated version with the demo section placed above:

# STS Project with Sentence-BERT

This project aims to solve the Semantic Text Similarity (STS) task using Sentence-BERT and cosine similarity as the loss function. The model is trained on the STS-B (cross-lingual) dataset, also known as stsb_multi_mt.

## Prerequisites

Before running the application, ensure that you have the following dependencies installed:

- Python 3.x
- pip (Python package installer)

## Demo

![predict_data](https://github.com/pnavin9/STS/assets/106406724/956c4a2c-89b0-4d47-a55b-50f540b7d464)



## Setup

1. Clone this repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd <project_directory>
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Decompress the pre-trained model file:

The pre-trained model file has been compressed to reduce its size. Please decompress the file before running the application. You can do this by running the following command:

```bash
tar -xf model.tar.gz
```

## Usage

To run the application, execute the following command:

```bash
python app.py
```

This will start the application, and you can access it through your web browser.

## Additional Information

### Dataset

The model is trained on the STS-B (cross-lingual) dataset, also known as stsb_multi_mt. This dataset contains sentence pairs in multiple languages along with their similarity scores. The model is trained to predict the similarity score between two sentences.

### Model

The model used in this project is Sentence-BERT (SBERT). Sentence-BERT is a modification of the popular BERT model that is fine-tuned for sentence-level tasks. It generates fixed-length sentence embeddings, which can be used to measure the similarity between sentences.

### Loss Function

The loss function used in this project is cosine similarity. Cosine similarity measures the cosine of the angle between two vectors and provides a similarity score between -1 and 1. A higher score indicates a higher similarity between the sentences.


## Contact

If you have any questions, feedback, or issues regarding this project, please feel free to reach out:

- Name: Navin Patwari
- Email: patwarinavin9@gmail.com
