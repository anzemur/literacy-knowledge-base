# Fictional characters analysis

While analysis of literary works and their content is a commonly taught and often simple skill used by people, it is a challenge for machines. They lack human knowledge, common sense, and contextual awareness, which is very important when analyzing literary works. Many researchers have tackled these problems, some more successfully than others. In our work, we approach the problems of character extraction, sentiment analysis of character relationships, and protagonist and antagonist detection. All of these tasks are performed on our newly created and annotated corpus of fables.

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
python -m spacy download en_core_web_trf
pip install allennlp-models
python src/downloads.py
```

## Running the code

### 1. Character recognition
To generate the results of character recognition you should run the following command:
```bash
python src/characters/run_ner.py
```
And to evaluate the obtain results you should run:
```bash
python src/eval_ner.py
```

### 2. Character sentiments
To generate the results of character sentiments & protagonist detection you should run the following command:
```bash
python src/characters/character_sentiments.py
```
And to evaluate the obtain results for character sentiments you should run:
```bash
python src/characters/eval_sentiments.py
```

### 2. Antagonist detection
To evaluate the obtain results for antagonist detection you should run:
```bash
python src/characters/eval_leads.py
```
