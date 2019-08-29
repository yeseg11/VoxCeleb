#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install librosa


# In[1]:


import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.layers import Input, GlobalAveragePooling2D, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Activation
from keras.models import Model
import decimal
import numpy
import math
import logging
import librosa
import numpy as np
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
from scipy.signal import lfilter, butter
from sklearn.model_selection import train_test_split


# # Prepare Training and Testing files

# In[2]:


filenames = ['./wav_id10270/PXmaB6Ui0fE/00001.wav',
             './wav_id10270/zjwijMp0Qyw/00001.wav',
             './wav_id10270/zjwijMp0Qyw/00002.wav',
             './wav_id10270/zjwijMp0Qyw/00003.wav',
             './wav_id10270/OmSWVqpb-N0/00001.wav',
             
             
            './wav_id10271/gO805KoL2RM/00001.wav',
             './wav_id10271/gO805KoL2RM/00002.wav',
             './wav_id10271/gO805KoL2RM/00003.wav',
             './wav_id10271/gO805KoL2RM/00004.wav',
             './wav_id10271/gO805KoL2RM/00005.wav',
             './wav_id10271/gO805KoL2RM/00006.wav',
             './wav_id10271/gO805KoL2RM/00007.wav',
            ]


speakers = [
    10270,
    10270,
    10270,
    10270,
    10270,
    
    
    10271,
    10271,
    10271,
    10271,
    10271,
    10271,
    10271,
    
]

data = pd.DataFrame(data = np.column_stack([filenames, speakers]) ,columns=['filename', 'speaker'])
X_train, X_test, y_train, y_test = train_test_split(data, data.speaker, test_size=0.33, random_state=42, stratify=data.speaker)
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)


# In[3]:


SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10


# In[4]:


COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)


# In[5]:


ENROLL_LIST_FILE = "./train.csv"
TEST_LIST_FILE = "./test.csv"
RESULT_FILE = "./results.csv"


# In[6]:


WEIGHTS_FILE = 'weights.h5'


# # Network

# In[7]:


def conv_bn_pool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
    pool='',pool_size=(2, 2),pool_strides=None,
    conv_layer_prefix='conv'):
    x = ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
    x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
    x = Activation('relu', name='relu{}'.format(layer_idx))(x)
    if pool == 'max':
        x = MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='mpool{}'.format(layer_idx))(x)
    elif pool == 'avg':
        x = AveragePooling2D(pool_size=pool_size,strides=pool_strides,name='apool{}'.format(layer_idx))(x)
    return x


def conv_bn_dynamic_apool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
    conv_layer_prefix='conv'):
    x = ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
    x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
    x = Activation('relu', name='relu{}'.format(layer_idx))(x)
    x = GlobalAveragePooling2D(name='gapool{}'.format(layer_idx))(x)
    x = Reshape((1,1,conv_filters),name='reshape{}'.format(layer_idx))(x)
    return x


def vggvox_model():
    inp = Input(INPUT_SHAPE,name='input')
    x = conv_bn_pool(inp,layer_idx=1,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1),
        pool='max',pool_size=(3,3),pool_strides=(2,2))
    x = conv_bn_pool(x,layer_idx=2,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1),
        pool='max',pool_size=(3,3),pool_strides=(2,2))
    x = conv_bn_pool(x,layer_idx=3,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    x = conv_bn_pool(x,layer_idx=4,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    x = conv_bn_pool(x,layer_idx=5,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1),
        pool='max',pool_size=(5,3),pool_strides=(3,2))		
    x = conv_bn_dynamic_apool(x,layer_idx=6,conv_filters=4096,conv_kernel_size=(9,1),conv_strides=(1,1),conv_pad=(0,0),
        conv_layer_prefix='fc')
    x = conv_bn_pool(x,layer_idx=7,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0),
        conv_layer_prefix='fc')
    x = Lambda(lambda y: K.l2_normalize(y, axis=3), name='norm')(x)
    x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid', name='fc8')(x)
    m = Model(inp, x, name='VGGVox')
    return m


# # Processing

# In[8]:


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win

def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,))):
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]

def magspec(frames, NFFT):
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)

def powspec(frames, NFFT):
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))

def logpowspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps

def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])


# # Reading .wav file

# In[9]:


def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

def normalize_frames(m,epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])

def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1,-1], [1,-alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout


def get_fft_spectrum(filename, buckets):
    signal = load_wav(filename,SAMPLE_RATE)
    signal *= 2**15

    signal = remove_dc_and_dither(signal, SAMPLE_RATE)
    signal = preemphasis(signal, coeff=PREEMPHASIS_ALPHA)
    frames = framesig(signal, frame_len=FRAME_LEN*SAMPLE_RATE, frame_step=FRAME_STEP*SAMPLE_RATE, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=NUM_FFT))
    fft_norm = normalize_frames(fft.T)

    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1]-rsize)/2)
    out = fft_norm[:,rstart:rstart+rsize]
    return out


# In[10]:


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5    enroll_result = get_embeddings_from_list_file(model, ENROLL_LIST_FILE, MAX_SEC)

        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets

def get_embeddings_from_list_file(model, list_file, max_sec):
    buckets = build_buckets(max_sec, BUCKET_STEP, FRAME_STEP)
    result = pd.read_csv(list_file, delimiter=",")
    result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
    result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
    return result[['filename','speaker','embedding']]

def get_id_result():
    print("Loading model weights from [{}]....".format(WEIGHTS_FILE))
    model = vggvox_model()
    model.load_weights(WEIGHTS_FILE)
    model.summary()

    print("Processing enroll samples....")
    enroll_result = get_embeddings_from_list_file(model, ENROLL_LIST_FILE, MAX_SEC)
    enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
    speakers = enroll_result['speaker']

    print("Processing test samples....")
    test_result = get_embeddings_from_list_file(model, TEST_LIST_FILE, MAX_SEC)
    test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

    print("Comparing test samples against enroll samples....")
    distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=COST_METRIC), columns=speakers)

    scores = pd.read_csv(TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
    scores = pd.concat([scores, distances],axis=1)
    scores['result'] = scores[speakers].idxmin(axis=1)
    scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int

    print("Writing outputs to [{}]....".format(RESULT_FILE))
    with open(RESULT_FILE, 'w') as f:
        scores.to_csv(f, index=False)


# In[ ]:





# # Training

# In[11]:


model = vggvox_model()
model.load_weights(WEIGHTS_FILE)


# In[12]:


model.compile(optimizer='adam', loss='mae')


# In[13]:


buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)
data = pd.read_csv(ENROLL_LIST_FILE, delimiter=",")
data['features'] = data['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
data.head()


# In[14]:


feats = data.features.values


# In[15]:


data_train = []
for each in feats:
    data_train.append(each.reshape(1,*each.shape,1))


# In[16]:


y_train = []
for each in feats:
    x = each.flatten()
    x = x.tolist()
    x = x[0:1024]
    x = np.array(x)
    y_train.append(x.reshape(1,1,1024))


# In[23]:


def generator():
    k = 0
    while True:
        if k >= len(data_train) or k >= len(y_train):
            k = 0
        yield (data_train[k], [[y_train[k]]])
        k += 1


# In[24]:


train_gen = generator()


# In[25]:


model.fit_generator(train_gen, steps_per_epoch=int(len(data_train)), epochs=50)


# In[26]:


get_id_result()


# # Results

# In[27]:


res = pd.read_csv('./results.csv')


# In[30]:


res


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




