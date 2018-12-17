from scipy import linalg
import numpy as np
from random import random, seed

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize
from scipy.misc import comb

def olsRegresjon(X_train, y_train, boolPrintKonfIntv):
    '''
    olsRegresjon:
        X_train, y_train: treningsdata for problemet y=X @ beta
        boolPrintKonfIntv: om konfidensintervalla skal verta printa

        ret: betaverdiar
    '''

    # svd løysing
    U,s,V=np.linalg.svd(X_train)
    D=np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(D, s)
    betas = V.T @ linalg.pinv(D) @ U.T @ y_train

    return betas

def ridgeRegresjon(X_train, y_train, straffeparam):
    '''
    ridgeRegresjon:
        X_train, y_train: treningsdata for problemet y=X @ beta
        straffeparam: straffeparameter som berre vert aktuell for ridge/lasso

        ret: betaverdiar
    '''

    # svd løysing
    U,s,V=np.linalg.svd(X_train)
    D=np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(D, s)
    core=linalg.pinv((D.T @ D)+straffeparam*np.eye(X_train.shape[1]))
    betas = V.T @ core @ D.T @ U.T @ y_train

    return betas


def lassoRegresjonScipy(X_train, y_train, straffeparam):
    '''
    lassoRegresjonScipy:
        X_train, y_train: treningsdata for problemet y=X @ beta
        straffeparam: straffeparameter som berre vert aktuell for ridge/lasso

        ret: betas
    '''
    lasso=linear_model.Lasso(alpha=straffeparam, fit_intercept=False)
    lasso.fit(X_train,y_train)
    return lasso.coef_

def kryssvalidering(X_train, X_test, y_train, y_test, regfunc, k, straffeparam=0):
    '''
    kryssvalidering:
        kryssvalidering gjev betaverdiane samt utrekingar om ein ønskjer statistikk.
        Funksjonen kryssvalidering åleine reknar altså ikkje ut statistikk,
        då ein ikkje alltid ønskjer statistikk. Om ein vil ha statistikk, sjå funksjonen kryssvalideringsTest.

        X_train, X_test, y_train, y_test: høvesvis trenings og testdata for modellen y= X @ betas
        regfunc: regresjonsfunksjon (ML, ridge ..)
        k: talet på oppdelingar på datamengda
        straffeparam: straffeparameter som berre vert aktuell for ridge/lasso

        ret: betaverdiar samt. div. resultat for vidare utreking av statistikk
    '''
    seglen=int(len(X_train)/k)

    # set opp arrays for betaverdiar, prediksjonar og ekte verdiar for seinare statistikkutrekningar
    betasarray = np.empty((k, X_train.shape[1])) #np.empty((k, np.int(comb(2+polygrad, polygrad))))
    zs_tests = np.empty((k, seglen))
    zs_hat_preds = np.empty((k, seglen))

    # det første segmentet av datamengda vert sett av til seinare utrekningar for bias/varians, dette er X_test, y_test

    # merk at eg likevel reknar ut for det første segmentet, men det vert utelate frå bias-variance dekomponeringa
    # meir detaljar i metodeseksjonen i rapporten
    for i in range(0,k):
        testIndices = np.arange(i*seglen, (i+1)*seglen)
        trenIndices = np.array([ind for ind in range(0, len(X_train)) if ind not in testIndices])

        segX_test, segy_test = (X_train[testIndices], y_train[testIndices])
        segX_tren, segy_tren = (X_train[trenIndices], y_train[trenIndices])

        if regfunc == olsRegresjon:
            betas = regfunc(segX_tren, segy_tren, False)
        elif regfunc == ridgeRegresjon or regfunc == lassoRegresjonScipy:
            betas = regfunc(segX_tren, segy_tren, straffeparam)

        zs_hat_preds[i]=predict(betas, segX_test)
        betasarray[i]=betas
        zs_tests[i]=segy_test

    zs_hat_preds_for_bias=[predict(betas, X_test) for betas in betasarray]

    return [betasarray, (zs_tests, zs_hat_preds), (zs_hat_preds_for_bias, y_test)]


def predict(betas, X_test):
    '''
    predict:
        Nær same funksjonalitet som predict i scipy.
        X_test: datamatrise for prediksjon
        betas: betaverdiane knytte til polynomet det vert gjort regresjon for
    '''
    return X_test @ betas

def getStatsFromResampling(zs, zs_hat_preds, polygrad, k):
    '''
    getStatsFromResampling:
        Gjev ymse statistikk med omsyn på resamplinga som er vorten gjord.
        Den er eigentleg berre ein versjon av getStatsFromRegression som handsamar
        fleirdimensjonelle arrays. Numpy si arrayhandsaming vart ikkje optimal nok til å berre ha ein funksjon her.

        zs: dei sampla zs-verdiane (dei reelle verdiane)
        zs_hat: resultatet frå ein regresjon på zs
        polygrad: graden på polynomet knytt til zs
    '''
    stats=np.array([getStatsFromRegression(zs[i], zs_hat_preds[i], polygrad)[:2] for i in range(k)])
    stats=1/len(stats)*stats.sum(axis=0)
    return np.append(stats,r2score(zs, zs_hat_preds, stats[0]))

def getStatsFromRegression(zs, zs_hat_preds, polygrad):
    '''
    getStatsFromRegression:
        Gjev ymse statistikk med omsyn på regresjonen som er vorten gjord

        zs: dei sampla zs-verdiane (dei reelle verdiane)
        zs_hat: resultatet frå ein regresjon på zs
        polygrad: graden på polynomet knytt til zs
    '''
    mean=1/len(zs)*np.sum(zs)
    mse=1/polygrad*np.sum((zs-zs_hat_preds)**2)
    return np.array([mean, mse, r2score(zs, zs_hat_preds, mean)])

def r2score(zs, zs_hat_preds, mean):
    return 1-np.sum((zs-zs_hat_preds)**2)/np.sum((zs-mean)**2)
