# Literacy situation models knowledge base creation

Building a knowledge base based on situation models from selected English/Slovene short stories. Knowledge base can focus on a subset of the following inference types: Referential, Case structure role  assignment, Causal antecedent, Superordinate goal, Thematic, Character emotional reaction, Causal consequence, Instantiation of noun category, Instrument, Subordinate goal, State, Emotion of reader, Author's intent.


## Dataset
Dataset is scrapped from the Project Gutenberg website which provides free eBooks, with the focus on older works for which U.S. copyright has expired. We limited our dataset to only english books with a public license so we ended up with 818 short stories.

To obtain the dataset run `./src/data_parsing/scrape_gutenberg.py` script.

