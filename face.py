'''
    File name: face.py
    Author: Gabriel Moreira
    Date last modified: 07/07/2022
    Python Version: 3.9.13
'''

import os
import json
import random
import numpy as np
from tqdm import tqdm 
from PIL import Image
from datetime import datetime
from datetime import timedelta

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from utils import *


class Faces:
    def __init__(self, ROOT, device):
        self.ROOT   = ROOT
        self.device = device
        self.sync_folders()
        
        self.BOUNDING_BX_PATH = os.path.join(ROOT, 'boxes')
        self.CROP_PATH        = os.path.join(ROOT, 'crops')
        self.JSON_PATH        = os.path.join(ROOT, 'data.json')

        self.bounding_bx_filenames = os.listdir(self.BOUNDING_BX_PATH)
        self.crop_filenames        = os.listdir(self.CROP_PATH)

        self.faces = []

        with open(os.path.join('retrieved', "data.json"), 'r') as fp:
            self.data_dict = json.load(fp)

        self.process_faces()


    def get_bounding_box(self, i):
        """
            Return Pillow bounding-box image of the i-th face in the list
        """
        img = Image.open(self.faces[i]['bounding_bx_filename'])
        return img


    def get_crop(self, i):
        """
            Return Pillow crop image of the i-th face in the list
        """
        img = Image.open(self.faces[i]['crop_filename'])
        return img


    def get_src_video_list(self):
        """
            Get all video source paths used to retrieve the photos.
            Use this to check if specific video has been processed
        """
        return sorted(list(self.data_dict.values()))


    def search(self, **kwargs):
        """
            Search for faces in the self.face list according to
            user-specified criteria
        """
        selection = {}
        for i, face in enumerate(self.faces):
            flags = [True]*4
            if 'src_hash' in kwargs.keys() and face['src_hash'] != kwargs['src_hash']:
                flags[0] = False 
            if 'src_name' in kwargs.keys() and face['src_name'] != kwargs['src_name']:
                flags[1] = False
            if 'from_date' in kwargs.keys() and face['time_src_end'] <= datetime.strptime(kwargs['from_date'], '%Y%d%m%H%M%S'):
                flags[2] = False
            if 'to_date' in kwargs.keys() and face['time_src_begin'] >= datetime.strptime(kwargs['to_date'], '%Y%d%m%H%M%S'):
                flags[3] = False

            if all(flags):
                selection[i] = face

        if len(selection) == 0:
            print('No face matches the search criteria')

        return selection


    def show_sample(self):
        """
            Plots random sample of 18 faces
        """
        plt.figure(figsize=(20,10))
        plt.title("{}/{} faces".format(18,len(self.faces)))
        idx = random.sample(range(len(self.faces)), 18)
        for i in range(18):
            plt.subplot(3,6,i+1)
            plt.title("#{}   {}".format(idx[i], self.faces[idx[i]]['time_src_begin'] + 
                timedelta(0,np.floor(self.faces[idx[i]]['frame_no']/25))), fontsize=11)
            plt.imshow(self.faces[idx[i]]['img'])
            plt.axis('off')


    def show_statistics(self):
        """
            Shows histogram of #faces over time
        """
        timestamps = []
        for face in self.faces:
            timestamps.append(face['time_src_begin'] + timedelta(0,np.floor(face['frame_no']/25)))

        start_time = min(timestamps)
        end_time   = max(timestamps)

        delta = timedelta(hours=2)

        bins = np.arange(start_time, end_time, delta).astype(datetime)
        counts = np.zeros((len(bins)))
        for timestamp in timestamps:
            for i in range(1,len(bins)):
                if timestamp >= bins[i-1] and timestamp < bins[i]:
                    counts[i-1] += 1

        plt.figure(figsize=(20,10))
        sns.set_theme()

        plt.title('When were the {} faces detected?'.format(len(self.faces)), fontsize=18)
        plt.plot(bins + timedelta(hours=2) / 2, counts, 'o-', color='mediumvioletred')
        plt.xlabel('Date (MM-DD HH)', fontsize=18);
        plt.ylabel('No. faces captured', fontsize=18);


    def sync_folders(self):
        """
            Eliminates files from crops folder which are not in
            the bounding box folder
        """
        BOUNDING_BX_PATH = os.path.join(self.ROOT, 'boxes')
        CROP_PATH        = os.path.join(self.ROOT, 'crops')

        bounding_bx_filenames = os.listdir(BOUNDING_BX_PATH)
        crop_filenames        = os.listdir(CROP_PATH)

        bounding_bx_hash_frames = [filename.split('_')[0] + '_' + 'frame_' + filename.split('_')[2].split('.')[0] for filename in bounding_bx_filenames]

        for cropped_face_file in crop_filenames:
            _, extension = os.path.splitext(cropped_face_file)

            if extension == '.jpg':
                crop_hash_frame = cropped_face_file.split('_')[0] + '_' + 'frame_' + cropped_face_file.split('_')[2]
                    
                if crop_hash_frame not in bounding_bx_hash_frames:
                    os.remove(os.path.join(CROP_PATH, cropped_face_file))
            else:
                os.remove(os.path.join(CROP_PATH, cropped_face_file))


    def process_faces(self):
        """
            Builds list of face dictionaries.
            To be called by the class constructor.
        """
        for filename in self.crop_filenames:
            hashed_path, frame_no, face_no = parse_filename(filename)

            if hashed_path in self.data_dict.keys():
                src_filename = self.data_dict[hashed_path]
                src_begin, src_end = get_timestamps(src_filename)

                face = {'img'                  : Image.open(os.path.join(self.CROP_PATH, filename)),
                        'src_hash'             : hashed_path,
                        'src_name'             : src_filename,
                        'frame_no'             : frame_no,
                        'face_no'              : face_no,
                        'time_src_begin'       : src_begin,
                        'time_src_end'         : src_end,
                        'crop_filename'        : os.path.join(self.CROP_PATH, filename),
                        'bounding_bx_filename' : os.path.join(self.BOUNDING_BX_PATH, hashed_path + "_frame_" + str(frame_no) + ".jpg")}

                self.faces.append(face)

        print("Ready. {} face images in the database.\n".format(len(self.faces)))


    def compute_features(self, model):
        """
            Get face embeddings and store them in the dictionary
            with the rest of the face attributes
        """
        model.eval()
        totensor = transforms.ToTensor()
        resize   = transforms.Resize((224,224))

        bar = tqdm(total=len(self.faces), dynamic_ncols=True, desc='Computing features') 
        for i in range(len(self.faces)): 
            with torch.no_grad():
                img_tensor = totensor(resize(self.faces[i]['img'])).to(self.device)
                features   = model(img_tensor.unsqueeze(0), return_features=True)
                features   = features.cpu().detach().numpy()
                self.faces[i]['features'] = features
            bar.update()
        bar.close()


    def __getitem__(self, i):
        """
            Override subscript [] operator
        """
        return self.faces[i]


    def __len__(self):
        """
            Override len() operator
        """
        return len(self.faces)
