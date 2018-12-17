import numpy as np
from scipy import signal
import glob

import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import sklearn.model_selection as skms
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_widths

paths = ['KreftsvulstVevEEG', 'KreftSvulstVanligVev', 'OyneLukketEEG', 'OyneOpenEEG', 'Seizure']
identPaths = ['F', 'N', 'O', 'Z', 'S']

# epileptisk anfall 1, ikkje anfall - 0
nCat = 5;
labels = np.zeros(100*23*nCat)
labels[100*(nCat-1)*23:]=1

chunksize = 178 # 23 chunks for kvar fil/person

allChunks = np.empty((nCat,100,23,178), object)

def loadNonSeizureData():
    '''
        loadNonSeizureData:
            lastar inn dei 11500 ulike signala frå datafilene

            ret: gjev datamatrisa med dei 11500 signala
    '''
    for path, id in zip(paths[:nCat], identPaths[:nCat]):
        for pid in ['%03d' % i for i in range(1,101)]:
            with open('Data/'+path+'/'+id+pid+'.txt') as f:
                lines = np.array(f.readlines(), dtype=np.int32)
                #lines = signal.cwt(lines, signal.ricker, np.arange(1,5)).flatten()

            chunks = np.array([lines[i:i+chunksize] for i in range(0, len(lines)-(len(lines) % chunksize), chunksize)]) # inneheld 23 oppdelingar

            allChunks[identPaths.index(id)][int(pid)-1] = chunks

    return allChunks.reshape((allChunks.shape[0]*allChunks.shape[1]*allChunks.shape[2],178))

data = loadNonSeizureData().astype(np.int32)

def featureExtraction(data, boolNormalizeData):
    '''
        featureExtraction:
            Gjer ei eigenskapsuttrekking med gjennomsnitt, standardavvik, kurtose, skeivskap, min og max verdiar
            frå signala gjevne i datamengda. Dette er etterfølgt av ein prinsipalkomponentanalyse som
            tek vare på 98 prosent av total varians.

                param:
                    data: datamengda
                    boolNormalizeData: bool som indikerer om datamengda skal normaliserast

            return: X_train, X_test, y_train, y_test
    '''
    feats = [np.mean, np.std, kurtosis, skew, np.min, np.max]
    nydata = np.empty((data.shape[0],len(feats)))
    for i in range(data.shape[0]):
        nydata[i] = [feat(data[i]) for feat in feats]


    pca = PCA(2) # gjev dei 2 komponentane gjev 98 prosent av total varians
    nydata = pca.fit_transform(nydata)

    if boolNormalizeData:
        nydata = (nydata-np.mean(nydata))/(np.var(nydata)**2) # normaliserer data

    return (skms.train_test_split(nydata, labels, test_size=0.2), nydata)

def loadFullSignals():
    '''
        loadFullSignals:
            Returnerer ei datamengde som representerer dei fulle signala, altså 4097 datapunkt og ei liste av faktisk klassifisering av desse signala.
            Vert nytta til einingstestar av koden samt RNN.

            return: X og y verdiar knytte til problemet y = X @ betas
    '''
    signals = np.empty((nCat,100,4097), object)
    for path, id in zip(paths[:nCat], identPaths[:nCat]):
        for pid in ['%03d' % i for i in range(1,101)]:
            with open('Data/'+path+'/'+id+pid+'.txt') as f:
                signals[identPaths.index(id)][int(pid)-1] = np.array(f.readlines(), dtype=np.int32)

    labels = np.zeros(100*nCat)
    labels[100*(nCat-1):]=1
    X=signals.reshape(100*nCat,4097)
    return X, labels

# einingstestar for datamengda - då koden for datamengda vart sett opp på eiga hand
def findFile(matchstr):
    '''
        findFile:
            Søkjer etter om matchstr finst i ei av filene i datamappa.
            Elles vert det gjeve ei feilmelding.

            parametrar:
                matchstr: strengen som skal verta funnen att
    '''
    funnen = False;
    for file in glob.glob('Data/*/*'):
        with open(file) as f:
            contents = f.read()
        contents = repr(contents)

        if matchstr in contents:
            print(file)
            funnen = True;
    assert funnen == True

def testDataSetup():
    '''
        testDataSetup:
            Einingstest av koden som sjekker om datamengda er riktig sett opp.
            Meir detaljert - det vert altså sjekka om informasjonen i datamengda vert funnen i rette filer.
    '''
    for i in range(0, len(data), 23):
        matchstr=repr(''.join(str(c)+str("\n") for c in data[i]))[1:200]
        findFile(matchstr)

# plot epilepsi/ikke-epilepsi signal
def plotEpilepsySignalVsOrdinarySignal():
    signals, labels = loadFullSignals()

    t=np.arange(0,26.6, 26.6/4097)

    plt.plot(t, np.array(signals[(nCat*4)]), label='Epileptisk hjerneaktivitet')
    plt.plot(t, np.array(signals[(nCat*1)]), label='Normal hjerneaktivitet')
    plt.legend()
    plt.show()

#plotEpilepsySignalVsOrdinarySignal()
