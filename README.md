# Fictional characters analysis

Building a knowledge base based on situation models from selected English/Slovene short stories. Knowledge base can focus on a subset of the following inference types: Referential, Case structure role  assignment, Causal antecedent, Superordinate goal, Thematic, Character emotional reaction, Causal consequence, Instantiation of noun category, Instrument, Subordinate goal, State, Emotion of reader, Author's intent.

## Dataset
Dataset is scrapped from the Project Gutenberg website which provides free eBooks, with the focus on older works for which U.S. copyright has expired. We decided to use a collection of fables by the greek author Aesop called [The Fables of Aesop](https://www.gutenberg.org/cache/epub/28/pg28.txt) collected and translated by Joseph Jacobs. We collected 55 of these fables and annotated them by hand. For each fable we annotated the following things:
* characters,
* sentiment relationships between the characters,
* protagonist and antagonist of the story.

You can find the dataset and the annotations in the following directory: `data/aesop/`. Annotations are saved in JSON format.



# Instructions

## Installation
1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or make sure that your [Python version is 3.8.x](https://www.python.org/downloads/). If you are using Anaconda you can create and activate new environment by running:

```bash
conda create -n <env_name> python=3.8
conda activate <env_name>
```


2. Clone this repository:
```bash
git clone https://github.com/anzemur/literacy-knowledge-base.git
```

3. Install dependencies:
```bash
pip install -r requirements.txt 
```

4. Download & install language models:
```bash
python -m spacy download en_core_web_lg
pip install allennlp-models
python src/downloads.py
```

## Running the code

### 1. Character recognition
To generate the results of character recognition you should run the following command:
```bash
python src/run_ner.py
```
And to evaluate the obtain results you should run:
```bash
python src/eval_ner.py
```

### 2. Character sentiments
To generate the results of character sentiments & protagonist detection you should run the following command:
```bash
python src/character_sentiments.py
```
And to evaluate the obtain results for character sentiments you should run:
```bash
python src/eval_sentiments.py
```

### 2. Antagonist detection
To evaluate the obtain results for antagonist detection you should run:
```bash
python src/eval_leads.py
```

No coref:
F1-score: 0.6730355503082776
Precision: 0.8712121212121212
Recall: 0.6412554112554114

Coref (first noun, no title):
F1-score: 0.7302164502164499
Precision: 0.8812121212121211
Recall: 0.6992640692640694

Coref (most common, no title)
F1-score: 0.7276623376623373
Precision: 0.8615151515151512
Recall: 0.7068398268398269

Coref (most common, title)
F1-score: 0.7021009293736563
Precision: 0.7557575757575757
Recall: 0.7232900432900432

Coref (First noun, title)
F1-score: 0.686646383919111
Precision: 0.7754545454545454
Recall: 0.7035930735930737