import xml.etree.ElementTree as ET
import itertools

tree = ET.parse('data/raw/1_1ecbplus.xml.xml')
root = tree.getroot()

print('root elems')
for x in root:
    print(x)

tokens = root.findall('token')

print('markables')
for x in root.findall('Markables')[0]:
    print(x)

entities = []

print('people')
# people    
for x in root.findall('Markables')[0].findall('HUMAN_PART_PER'):
    print(x)
    id_text = ''
    if len(x) == 0:
        print('\t>>> no tokens, skipped')
        continue
    for x_token in x:
        for token in tokens:
            if token.attrib['t_id'] == x_token.attrib['t_id']:
                id_text += ' ' + token.text
                break
    print(f'\t{id_text}')
    entities.append((x.attrib['m_id'], 'PER', id_text))

# organisations
print('organisations')
for x in root.findall('Markables')[0].findall('HUMAN_PART_ORG'):
    print(x)
    id_text = ''
    if len(x) == 0:
        print('\t>>> no tokens, skipped')
        continue
    for x_token in x:
        for token in tokens:
            if token.attrib['t_id'] == x_token.attrib['t_id']:
                id_text += ' ' + token.text
                break
    print(f'\t{id_text}')
    entities.append((x.attrib['m_id'], 'ORG', id_text))

print('locations')
# locations    
for x in root.findall('Markables')[0].findall('LOC_FAC'):
    print(x)
    id_text = ''
    if len(x) == 0:
        print('\t>>> no tokens, skipped')
        continue
    for x_token in x:
        for token in tokens:
            if token.attrib['t_id'] == x_token.attrib['t_id']:
                id_text += ' ' + token.text
                break
    print(f'\t{id_text}')
    entities.append((x.attrib['m_id'], 'LOC', id_text))

action_ids = []
actions = []
print('actions')
# actions    
for x in itertools.chain(
    root.findall('Markables')[0].findall('ACTION_ASPECTUAL'),
    root.findall('Markables')[0].findall('ACTION_OCCURRENCE'),
    root.findall('Markables')[0].findall('ACTION_REPORTING'),
    root.findall('Markables')[0].findall('ACTION_STATE')
    ):
    print(x)
    id_text = ''
    if len(x) == 0:
        print('\t>>> no tokens, skipped')
        continue
    for x_token in x:
        for token in tokens:
            if token.attrib['t_id'] == x_token.attrib['t_id']:
                id_text += ' ' + token.text
                break
    print(f'\t{id_text}')
    climaxEvent = None
    if 'climaxEvent' in x.attrib:
        climaxEvent = x.attrib['climaxEvent']
    action_ids.append(x.attrib['m_id'])
    actions.append((x.attrib['m_id'], x.tag, id_text, climaxEvent))

target_count = 0
target_miss = 0
source_count = 0
source_miss = 0

links = []
print('links')
# links    
for x in root.findall('Relations')[0].findall('PLOT_LINK'):
    print(x)
    if len(x) == 0:
        print('\t>>> no tokens, skipped')
        continue
    source = x.find('source').attrib['m_id']
    target = x.find('target').attrib['m_id']
    if source in action_ids:
        source_count += 1
    else:
        source_miss += 1
        print('source miss', source)
    if target in action_ids:
        target_count += 1
    else:
        target_miss += 1
    print(f'\t{source} -> {target}')
    links.append((x.attrib['r_id'], x.attrib['relType'], source, target))

print('source actions', source_count, source_miss)
print('target actions', target_count, target_miss)