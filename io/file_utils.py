import re

def parse_ids_from_filename(filename):
    """
    从文件名解析 wafer_id, reticle_id, row_id, C_index, die
    例：'MKW_STB3SP9W00_R3C3_L_f7.48MHz_...' → wafer='STB3SP9W00', reticle='R3C3', row='R3', c=3, die='L'
    """
    die_match = re.search(r'(STB\w+_R\d+C\d+_([A-Z]\d*))', filename)
    if not die_match:
        return None, None, None, None, None
    die_id = die_match.group(1)
    die = die_match.group(2)
    wafer_id = die_id.split('_')[0]
    rc_match = re.search(r'(R\d+)C(\d+)', die_id)
    if not rc_match:
        return None, None, None, None, None
    row_id = rc_match.group(1)
    c_index = int(rc_match.group(2))
    reticle_id = f'{row_id}C{c_index}'
    return wafer_id, reticle_id, row_id, c_index, die

def parse_frequency_from_filename(filename):
    """从文件名解析频率（MHz），如 'f7.48MHz' → 7.48"""
    match = re.search(r'f([\d\.]+)MHz', filename)
    if match:
        return float(match.group(1))
    return None
