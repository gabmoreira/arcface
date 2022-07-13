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
from datetime import datetime, timedelta

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from utils import *
from metrics import *

class FaceLoader:
    def __init__(self, ROOT, device):
        self.ROOT   = ROOT
        self.device = device
        self.sync_folders()
        
        self.BOUNDING_BX_PATH = os.path.join(ROOT, 'boxes')
        self.CROP_PATH        = os.path.join(ROOT, 'crops')
        self.JSON_PATH        = os.path.join(ROOT, 'data.json')

        self.bounding_bx_filenames = os.listdir(self.BOUNDING_BX_PATH)
        self.crop_filenames        = os.listdir(self.CROP_PATH)

        self.faces    = []
        self.features = None

        with open(os.path.join('retrieved', "data.json"), 'r') as fp:
            self.data_dict = json.load(fp)

        self.process_faces()


    def show(self, index):
        plt.figure(figsize=(24,8))
        plt.subplot(1,2,1)
        plt.imshow(Image.open(self.faces[index]['crop_filename']))
        plt.title("ID {}".format(index))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(Image.open(self.faces[index]['bounding_bx_filename']))
        plt.title("Source: {}".format(self.data_dict[self.faces[index]['src_hash']]))
        plt.axis('off')


    def get_bounding_box(self, **kwargs):
        """
            Return Pillow bounding-box image
        """
        assert len(kwargs) == 1
        assert 'index' in kwargs.keys() or 'filename' in kwargs.keys()
        if 'index' in kwargs.keys():
            img = Image.open(self.faces[kwargs['index']]['bounding_bx_filename'])
        elif 'filename' in kwargs.keys():
            img = Image.open(kwargs['filename'])
        return img


    def get_crop(self, **kwargs):
        """
            Return Pillow crop image
        """
        assert len(kwargs) == 1
        assert 'index' in kwargs.keys() or 'filename' in kwargs.keys()
        if 'index' in kwargs.keys():
            img = Image.open(self.faces[kwargs['index']]['crop_filename'])
        elif 'filename' in kwargs.keys():
            img = Image.open(kwargs['filename'])
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
        selection = []
        for i, face in enumerate(self.faces):
            flags = [True]*4
            if 'src_hash' in kwargs.keys() and face['src_hash'] != kwargs['src_hash']:
                flags[0] = False 
            if 'src_name' in kwargs.keys() and face['src_name'] != kwargs['src_name']:
                flags[1] = False
            if 'from_date' in kwargs.keys() and face['time_src_end'] <= datetime.strptime(kwargs['from_date'], '%Y%m%d%H%M%S'):
                flags[2] = False
            if 'to_date' in kwargs.keys() and face['time_src_begin'] >= datetime.strptime(kwargs['to_date'], '%Y%m%d%H%M%S'):
                flags[3] = False

            if all(flags):
                selection.append(i)

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

                face = {'img'                  : self.nodistort_resize(Image.open(os.path.join(self.CROP_PATH, filename))),
                        'src_hash'             : hashed_path,
                        'src_name'             : src_filename,
                        'frame_no'             : frame_no,
                        'face_no'              : face_no,
                        'time_src_begin'       : src_begin,
                        'time_src_end'         : src_end,
                        'crop_filename'        : os.path.join(self.CROP_PATH, filename),
                        'bounding_bx_filename' : os.path.join(self.BOUNDING_BX_PATH, hashed_path + "_frame_" + str(frame_no) + ".jpg")}

                self.faces.append(face)

        print("{} face images available\n".format(len(self.faces)))


    def nodistort_resize(self, im, new_size=240):
        """
        """

        new_im = np.zeros([new_size,new_size,3])

        d = np.argmax(im.size)
        scaling = new_size / im.size[d]
        effective_size      = [0,0]
        effective_size[d]   = new_size
        effective_size[1-d] = int(np.floor(scaling * im.size[1-d]))
        margin = (240 - effective_size[1-d]) // 2
        if d == 0:
            new_im[margin:margin+effective_size[1-d],:,:] += np.array(im.resize(effective_size))
        elif d == 1:
            new_im[:,margin:margin+effective_size[1-d],:] += np.array(im.resize(effective_size))
        new_im = np.uint8(new_im)
        new_im = Image.fromarray(new_im)

        return new_im


    def compute_features(self, model):
        """
            Get face embeddings and store them in the dictionary
            with the rest of the face attributes
        """
        self.features = np.zeros((512, len(self.faces)))

        model.eval()
        totensor = transforms.ToTensor()
        resize   = transforms.Resize((224,224))
        flip     = transforms.RandomHorizontalFlip(1.0)

        bar = tqdm(total=len(self.faces), dynamic_ncols=True, desc='Computing features') 
        for i in range(len(self.faces)): 
            with torch.no_grad():
                img_tensor = totensor(resize(self.faces[i]['img'])).to(self.device)
                feat_r     = model(img_tensor.unsqueeze(0), return_features=True)
                feat_r     = feat_r.cpu().detach().numpy()
                feat_l     = model(flip(img_tensor).unsqueeze(0), return_features=True)
                feat_l     = feat_l.cpu().detach().numpy()
                feat       = 0.5*(feat_r + feat_l)
                self.features[:,i] += feat.reshape(512)
            bar.update()
        bar.close()


    def compute_similarities(self):
        self.sim = (self.features / np.linalg.norm(self.features,axis=0)).T @ (self.features / np.linalg.norm(self.features,axis=0))

        bar = tqdm(total=len(self.faces), dynamic_ncols=True, desc='Computing similarity') 
        for i in range(len(self.faces)):
            self.sim[i,i] = 0.0
            for j in range(i+1,len(self.faces)):
                if self.faces[i]['src_hash'] == self.faces[j]['src_hash']:
                    if abs(self.faces[i]['frame_no'] - self.faces[j]['frame_no']) < 500:
                        self.sim[i,j] = min(self.sim[i,j]+0.05, 1.0)
                        self.sim[j,i] = min(self.sim[j,i]+0.05, 1.0)
            bar.update()
        bar.close()


    def find_matches(self, index, num_matches):
        """
        """
        similarity = self.sim[index,:]
        similarity[index] = 0
        idx = np.argsort(-similarity)
        return idx[:num_matches], similarity[idx[:num_matches]]


    def show_matches(self, index, num_matches):
        """
        """
        idx, similarity = self.find_matches(index, num_matches)

        plt.figure(figsize=(30,15))
        plt.subplot(num_matches // 10, 10, 1)
        plt.imshow(self.faces[index]['img'])
        plt.title("Query", fontsize=20)
        plt.axis('off')

        for k in range(1, 10*(num_matches // 10) ):
            plt.subplot(num_matches // 10, 10, k+1)
            plt.imshow(self.faces[idx[k]]['img'])
            title = "#{}   {:.0f}% \n {}".format(idx[k], similarity[k]*100, 
            self.faces[idx[k]]['time_src_begin'] + timedelta(0,np.floor(self.faces[idx[k]]['frame_no']/25)))
            plt.title(title, fontsize=9)
            plt.axis('off')


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
