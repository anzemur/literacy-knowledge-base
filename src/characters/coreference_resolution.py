from allennlp.predictors.predictor import Predictor
import spacy
from utils import most_frequent

MODEL_URL = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'


def get_span_noun_indices(doc, cluster):
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]

    span_noun_indices = [i for i, span_pos in enumerate(spans_pos) if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices


def core_logic_part(document, coref, resolved, mention_span):
    final_token = document[coref[1]]

    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text.title() + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text.title() + final_token.whitespace_

    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""

    return resolved


def get_cluster_head(doc, cluster, noun_indices):
    noun_words = []
    for x in noun_indices:
        head_start, head_end = cluster[x]
        noun_words.append(doc[head_start:head_end+1].text.lower())

    head_idx = noun_indices[noun_words.index(most_frequent(noun_words))]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]

    print(noun_words)
    print("##########")

    return head_span, [head_start, head_end]


def is_containing_other_spans(span, all_spans):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


def coreference_resolution(short_story):
    nlp = spacy.load('en_core_web_lg')
    predictor = Predictor.from_path(MODEL_URL)

    clusters = predictor.predict(short_story)['clusters']
    doc = nlp(short_story)

    return replace_corefs(doc, clusters)
