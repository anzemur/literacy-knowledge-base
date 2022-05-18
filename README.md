# Literacy situation models knowledge base creation

Building a knowledge base based on situation models from selected English/Slovene short stories. Knowledge base can focus on a subset of the following inference types: Referential, Case structure role  assignment, Causal antecedent, Superordinate goal, Thematic, Character emotional reaction, Causal consequence, Instantiation of noun category, Instrument, Subordinate goal, State, Emotion of reader, Author's intent.


## Dataset
Dataset is scrapped from the Project Gutenberg website which provides free eBooks, with the focus on older works for which U.S. copyright has expired. We limited our dataset to only english books with a public license so we ended up with 818 short stories.

To obtain the dataset run `./src/data_parsing/scrape_gutenberg.py` script.



# Running the code

## Installation
1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or make sure that your [Python version is 3.8.x](https://www.python.org/downloads/). If you are using Anaconda you can create and activate new environment by running:

```bash
conda create -n nlp python=3.8
conda activate nlp
```

2. Install dependencies by running:
```bash
pip install -r requirements.txt 
```

3. Download & Install language models:
```bash
python -m spacy download en_core_web_lg
```
```bash
pip install allennlp-models
```
python downloads.py
