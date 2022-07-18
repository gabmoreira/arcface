'''
    File name: utils.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import torch
import os
import pandas as pd

class Tracker:
    def __init__(self, metrics, filename, load=False):
        '''
        '''
        self.filename = os.path.join(filename, 'tracker.csv')

        if load:
            self.metrics_dict = self.load()
        else:        
            self.metrics_dict = {}
            for metric in metrics:
                self.metrics_dict[metric] = []


    def update(self, **args):
        '''
        '''
        for metric in args.keys():
            assert(metric in self.metrics_dict.keys())
            self.metrics_dict[metric].append(args[metric])

        self.save()


    def isLarger(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[-1] < value


    def isSmaller(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[0] > value


    def save(self):
        '''
        '''
        df = pd.DataFrame.from_dict(self.metrics_dict)
        df = df.set_index('epoch')
        df.to_csv(self.filename)


    def load(self):
        '''
        '''
        df = pd.read_csv(self.filename)  
        metrics_dict = df.to_dict(orient='list')
        return metrics_dict


    def __len__(self):
        '''
        '''
        return len(self.metrics_dict)


def getNumTrainableParams(network):
    '''
    '''
    num_trainable_parameters = 0
    for p in network.parameters():
        num_trainable_parameters += p.numel()
    return num_trainable_parameters



def initWeights(m):
    '''
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


'''
if __name__ == '__main__':
    tracker = Tracker(['epoch', 'acc', 'loss'], 'tracker')

    tracker.update(epoch=1, acc=0,   loss=0)
    tracker.update(epoch=2, acc=1.0, loss=2.3)
    tracker.update(epoch=3, acc=1.0, loss=2.2)

    print(tracker.metrics_dict)

    print(tracker.isLarger('acc', 0.9))

    print('loaded traacker')
    t2 = Tracker(None, 'tracker', True)
    print(t2.metrics_dict)
'''