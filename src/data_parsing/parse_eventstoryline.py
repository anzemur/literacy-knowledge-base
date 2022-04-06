import xml.etree.ElementTree as ET
import itertools

file_source = 'data/raw/1_1ecbplus.xml.xml'
file_destination = 'data/net/1_1ecbplus'

def parse_eventstoryline(source, destination):
    tree = ET.parse(source)
    root = tree.getroot()
    print(f'processing {source}')

    # print('root elems')
    # for x in root:
        # print(x)

    tokens = root.findall('token')

    # print('markables')
    # for x in root.findall('Markables')[0]:
        # print(x)

    entities = []

    # print('people')
    # people    
    for x in root.findall('Markables')[0].findall('HUMAN_PART_PER'):
        # print(x)
        id_text = ''
        if len(x) == 0:
            # print('\t>>> no tokens, skipped')
            continue
        for x_token in x:
            for token in tokens:
                if token.attrib['t_id'] == x_token.attrib['t_id']:
                    id_text += ' ' + token.text
                    break
        # print(f'\t{id_text}')
        entities.append((x.attrib['m_id'], 'PER', id_text))

    # organisations
    # print('organisations')
    for x in root.findall('Markables')[0].findall('HUMAN_PART_ORG'):
        # print(x)
        id_text = ''
        if len(x) == 0:
            # print('\t>>> no tokens, skipped')
            continue
        for x_token in x:
            for token in tokens:
                if token.attrib['t_id'] == x_token.attrib['t_id']:
                    id_text += ' ' + token.text
                    break
        # print(f'\t{id_text}')
        entities.append((x.attrib['m_id'], 'ORG', id_text))

    # print('locations')
    # locations    
    for x in root.findall('Markables')[0].findall('LOC_FAC'):
        # print(x)
        id_text = ''
        if len(x) == 0:
            # print('\t>>> no tokens, skipped')
            continue
        for x_token in x:
            for token in tokens:
                if token.attrib['t_id'] == x_token.attrib['t_id']:
                    id_text += ' ' + token.text
                    break
        # print(f'\t{id_text}')
        entities.append((x.attrib['m_id'], 'LOC', id_text))

    action_ids = []
    actions = []
    # print('actions')
    # actions    
    for x in itertools.chain(
        root.findall('Markables')[0].findall('ACTION_ASPECTUAL'),
        root.findall('Markables')[0].findall('ACTION_OCCURRENCE'),
        root.findall('Markables')[0].findall('ACTION_REPORTING'),
        root.findall('Markables')[0].findall('ACTION_STATE'),
        root.findall('Markables')[0].findall('ACTION_PERCEPTION'),
        root.findall('Markables')[0].findall('NEG_ACTION_STATE'),
        root.findall('Markables')[0].findall('NEG_ACTION_ASPECTUAL'),
        root.findall('Markables')[0].findall('NEG_ACTION_OCCURRENCE'),
        root.findall('Markables')[0].findall('NEG_ACTION_REPORTING'),
        root.findall('Markables')[0].findall('NEG_ACTION_PERCEPTION')
        ):
        # print(x)
        id_text = ''
        if len(x) == 0:
            # print('\t>>> no tokens, skipped')
            continue
        for x_token in x:
            for token in tokens:
                if token.attrib['t_id'] == x_token.attrib['t_id']:
                    id_text += ' ' + token.text
                    break
        id_text = id_text.strip()
        # print(f'\t{id_text}')
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
    # print('links')
    # links    
    for x in root.findall('Relations')[0].findall('PLOT_LINK'):
        # print(x)
        if len(x) == 0:
            # print('\t>>> no tokens, skipped')
            continue
        source = x.find('source').attrib['m_id']
        target = x.find('target').attrib['m_id']
        if source in action_ids:
            source_count += 1
        else:
            source_miss += 1
            # print('source miss', source)
        if target in action_ids:
            target_count += 1
        else:
            target_miss += 1

        relType = None
        if 'relType' in x.attrib:
            relType = x.attrib['relType']
        # print(f'\t{source} -> {target}')
        links.append((x.attrib['r_id'], relType, source, target))

    # print('source actions', source_count, source_miss)
    # print('target actions', target_count, target_miss)
    # 4 hashtags, last one empty
    # hash, id, data
    # empty hash
    # ids for edges

    f = open(destination, 'w+')

    f.write(f'# a network\n')
    f.write(f'# {len(actions)} nodes & {len(links)} edges\n')
    f.write(f'# by struggling students\n')
    f.write(f'#\n')

    action_map = {}
    print(f'\tfound {len(actions)} events')
    for i in range(len(actions)):
        action_map[actions[i][0]] = i+1
        f.write(f'# {i+1} {actions[i][0]} "{actions[i][2]}" {actions[i][1]} {actions[i][3]}\n')

    f.write(f'#\n')

    print(f'\tfound {len(links)} links')
    for i in range(len(links)):
        f.write(f'{action_map[links[i][2]]} {action_map[links[i][3]]} {links[i][1]} {links[i][0]}\n')

    f.close()

# parse_eventstoryline(file_source, file_destination)
files = [
    '1_1ecbplus.xml.xml',
    '12_1ecbplus.xml.xml',
    '13_4ecbplus.xml.xml',
    '14_1ecbplus.xml.xml',
    '16_1ecbplus.xml.xml',
    '18_1ecbplus.xml.xml',
    '19_1ecbplus.xml.xml',
    '20_1ecbplus.xml.xml',
    '22_1ecbplus.xml.xml',
    '23_1ecbplus.xml.xml',
    '24_1ecbplus.xml.xml',
    '3_1ecbplus.xml.xml',
    '30_1ecbplus.xml.xml',
    '32_1ecbplus.xml.xml',
    '33_1ecbplus.xml.xml',
    '35_1ecbplus.xml.xml',
    '37_1ecbplus.xml.xml',
    '4_1ecbplus.xml.xml',
    '41_1ecbplus.xml.xml',
    '5_1ecbplus.xml.xml',
    '7_1ecbplus.xml.xml',
    '8_1ecbplus.xml.xml',
]

for file in files:
    base = file.split('.')[0]
    parse_eventstoryline(f'data/raw/{file}', f'data/net/{base}')