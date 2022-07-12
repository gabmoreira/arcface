'''
    File name: misc.py
    Author: Gabriel Moreira
    Date last modified: 07/07/2022
    Python Version: 3.9.13
'''

import os
from datetime import datetime

def sync_folders(root='./retrieved', bounding_bx_folder='boxes', crops_folder='crops'):
    """
        Syncs files between crops and bounding box folders
        We can delete the false positives from the bounding box imgs and then 
        call this to update crops
    """
    BOUNDING_BX_PATH = os.path.join(root, bounding_bx_folder)
    CROPS_PATH       = os.path.join(root, crops_folder)

    bounding_bx_files = os.listdir(BOUNDING_BX_PATH)
    crop_files        = os.listdir(CROPS_PATH)

    bounding_bx_hash_frames = [filename.split('_')[0] + '_' + 'frame_' + filename.split('_')[2].split('.')[0] for filename in bounding_bx_files]

    for cropped_face_file in crop_files:
        _, extension = os.path.splitext(cropped_face_file)

        if extension == '.jpg':
            crop_hash_frame = cropped_face_file.split('_')[0] + '_' + 'frame_' + cropped_face_file.split('_')[2]
            
            if crop_hash_frame not in bounding_bx_hash_frames:
                os.remove(os.path.join(CROPS_PATH, cropped_face_file))
        else:
            os.remove(os.path.join(CROPS_PATH, cropped_face_file))


def parse_filename(filename):
    """
    """
    elements    = filename.split('_')
    source_hash = elements[0]

    if len(elements) == 3:
        frame_no =int(elements[2].split('.')[0])
        face_no  = None
    elif len(elements) == 4:
        frame_no = int(elements[2])
        face_no  = int(elements[3].split('.')[0])
    else:
        raise ValueError("Unknown filename {}".format(filename))
        
    return source_hash, frame_no, face_no


def get_timestamps(filename):
    """
    """
    timestamps = []
    for p in filename.split('_'):
        if p[:4] == "2022":
            dt = datetime.strptime(p[:14], '%Y%m%d%H%M%S')
            timestamps.append(dt)
    assert len(timestamps) >= 1
    
    begin = min(timestamps)
    end   = max(timestamps) if len(timestamps) >= 1 else None
    return begin, end