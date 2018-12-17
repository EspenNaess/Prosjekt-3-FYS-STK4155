import warnings
from Regresjonsmetodar import *
from LastData import *
from NevraltNett import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier # til samanlikning med eiga implementering

def accuracy(y, y_hat):
    '''
    accuracy:
        param:
            - y, y_hat: forventa resultat og predikert resultat. Går ut frå at dei begge er numpyarrayar.

        ret:
            - mål på accuracy
    '''
    return 1/len(y)*np.sum(y == y_hat)

def findOptimalParam(cost, egenImpBool, learningrate_low, learningrate_high, tal_lag_low, tal_lag_high, n_cells_low, n_cells_high, epoch_low, epoch_high, batchstorleikar, n_cycles):
    '''
        findOptimalParam:
            Gjer eit tilfeldig søk etter best score basert på intervalla gjevne til metoden

                param:
                    cost: kostfunksjon, verdiar: 'standard'/'crossentropy'
                    egenImpBool: True om egen implementering skal testes, False for ski-kit.
                    learningrate_low, learningrate_high: intervall for læringsrate
                    tal_lag_low, tal_lag_high: intervall for talet på lag i modellen
                    n_cells_low, n_cells_high: intervall for talet på celler i eit lag
                    epoch_low, epoch_high: epokeintervall
                    batchstorleikar: ein array av moglege storleikar på batches
                    n_cycles: talet på syklar/forsøk som skal køyrast

            ret: optimale verdiar
    '''
    n_numbers = 5 # testar for 0.01,0.02,0.03,0.04,0.05 for alle desimaltal gjevne av learningrate_low, learningrate_high
    learningrates = np.array([np.logspace(learningrate_low, learningrate_high, np.abs(learningrate_low))*i for i in range(1,n_numbers)]).flatten()
    optscore, optlr, optn_lag, optn_celler = (0, 0, 0, 0)
    for i in range(n_cycles):
        lr= np.random.choice(learningrates)
        str_batch = np.random.choice(batchstorleikar)
        n_lag=np.random.randint(low=tal_lag_low, high=tal_lag_high)
        epokar=np.random.randint(low=epoch_low, high=epoch_high)
        n_lag_celler=np.random.randint(low=n_cells_low, high=n_cells_high, size=n_lag)

        if egenImpBool: # eiga implementering
            nett2 = NevraltNett(X_train, y_train, n_lag_celler, 'k', cost)
            nett2.trenNettverk(str_batch, lr, epokar)
            y=nett2.feedforward(X_test).flatten()
            y=np.int8(np.round(y))
        else:

            ''' # keras parametersøk, for treigt til å vera særleg nyttig
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(n_lag_celler[0], activation='softmax', input_dim=X_train.shape[1]))
            model.add(tf.keras.layers.Dropout(0.3))
            for i in range(1,len(n_lag_celler)):
                model.add(tf.keras.layers.Dense(n_lag_celler[i], activation='softmax')) # Dense - betyr fully connected layer, tar antall nerveceller, aktiveringsfunksjoen (lineær fra 0 til infty)
                model.add(tf.keras.layers.Dropout(0.3))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam er best
            model.fit(X_train, y_train, epochs=epokar, batch_size=str_batch, validation_data=[X_test, y_test], verbose=0)
            scoreres=model.evaluate(X_test, y_test)[1]
            '''

            mlp = MLPClassifier(solver              = 'sgd',      # Stochastic gradient descent.
                                activation          = 'logistic', # Skl name for sigmoid.
                                alpha               = 0.0,        # No regularization for simplicity.
                                hidden_layer_sizes  = n_lag_celler,
                                batch_size = str_batch,
                                max_iter = epokar,
                                learning_rate = 'constant',
                                #nesterovs_momentum = False,
                                learning_rate_init = lr)    # Full network is of size (1,3,3,1).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p=1 # prosent av datamengda
                mlp.fit(X_train[:int(p*len(X_train))],y_train[:int(p*len(y_train))])
            y=mlp.predict(X_test)

        scoreres=accuracy(y_test, y)
        print("Prøvde, gav score:{}, lr: {}, n_lag: {}, n_celler: {}, batchstr: {}, epokar: {}".format(scoreres,lr,n_lag, n_lag_celler, str_batch, epokar))

        if scoreres < 1 and scoreres > optscore:
            optscore, optlr, optn_lag, optn_celler = (scoreres, lr, n_lag, n_lag_celler)
            print("Beste score fram til nå :{}".format(scoreres))

    return optscore, optlr,optn_lag, optn_celler

(X_train, X_test, y_train, y_test), data = featureExtraction(data, False)

# testdøme med utvalde parametrar som har gjeve gode resultat
p=1
nett2 = NevraltNett(X_train[:int(p*len(X_train))], y_train[:int(p*len(y_train))], [521, 6], 'k', 'standard')
nett2.trenNettverk(32, 0.04, 136)
y=nett2.feedforward(X_test).flatten()
y=np.int8(np.round(y))
scoreres=accuracy(y_test, y)
print("Grannsemd :{}".format(scoreres))

# søk etter optimale parametrar

#optr2score, optlr,optn_lag, optn_celler = findOptimalParam(standard',False,-4, -2, 1, 2, 1, 1000, 100, 200, [32, 64, 128], 100) # 16, 32, 64, 128, 256
#print("Optimal score: {} med lr: {}, n_lag: {}, n_celler: {}".format(optr2score, optlr,optn_lag, optn_celler))

#optr2score, optlr,optn_lag, optn_celler = findOptimalParam('standard',False,-4, -1, 1, 3, 1, 1000, 100, 200, [32, 64, 128], 100) # 16, 32, 64, 128, 256
#print("Optimal score: {} med lr: {}, n_lag: {}, n_celler: {}".format(optr2score, optlr,optn_lag, optn_celler))

# døme med MLPClassifier
'''
p = 1 # prosent ein vil nytta av den opprinnelege treningsmengda

mlp = MLPClassifier(solver              = 'sgd',      # Stochastic gradient descent.
                    activation          = 'logistic', # Skl name for sigmoid.
                    alpha               = 0,        # No regularization for simplicity.
                    hidden_layer_sizes  = [429],
                    batch_size = 64,
                    max_iter = 178,
                    learning_rate = 'constant',
                    learning_rate_init = 0.3)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp.fit(X_train[:int(p*len(X_train))],y_train[:int(p*len(y_train))])

y=mlp.predict(X_test)
scoreres=accuracy(y_test, y)
print("Accuracy: {}:".format(scoreres))
'''

# kode for plott
# mykje her gav for stusselege resultat til presentasjon i rapport
def plotScoreForEpochs(scorefunc, epoch_low, epoch_high, pts):
    '''
        plotScoreForEpochs:
            Lagar eit plott over kor godt eit nevralt nettverk gjer det med omsyn til eit mål gjeve av scorefunc
            over ulike epokar.

            param:
                scorefunc: anten r2-score eller accuracy
                epoch_low, epoch_high: epokeintervall
                pts: talet på evaluerte punkt for plottet
    '''
    xpts = [pt for pt in range(epoch_low, epoch_high, int(epoch_high/pts))]
    scores = np.empty(0)
    for i in xpts:
        nett2 = NevraltNett(X_train, y_train, [521, 6], 'k', 'standard')
        nett2.trenNettverk(32, 0.04, 136)
        y=nett2.feedforward(X_test).flatten()
        y=np.int8(np.round(y))
        scores=np.append(scores, scorefunc(y_test, y))
        print("Ferdig med {} av {} iterasjonar".format(i+1, len(xpts)))

    plt.plot(xpts, scores)
    plt.legend()
    #plt.xscale('log')
    plt.xlabel('Epokar')
    plt.ylabel('Grannsemd (accuracy)')
    plt.show()

#plotScoreForEpochs(accuracy, 10, 150, 15)

# test med ulike straffeparametrar, vart for fåfengt resultat til presentasjon i rapport
'''
alphas = [10**(i) for i in range(-7, 2)]
scores = np.empty(len(alphas))
for alpha in alphas:
    mlp = MLPClassifier(solver              = 'sgd',      # Stochastic gradient descent.
                        activation          = 'relu', # Skl name for sigmoid.
                        alpha               = alpha,        # No regularization for simplicity.
                        hidden_layer_sizes  = [521, 6],
                        batch_size = 32,
                        max_iter = 136,
                        learning_rate = 'constant',
                        learning_rate_init = 0.04)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mlp.fit(X_train[:int(p*len(X_train))],y_train[:int(p*len(y_train))])

    y=mlp.predict(X_test)
    scores[alphas.index(alpha)] = accuracy(y_test, y)

plt.plot(alphas, scores)
plt.legend()
#plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Grannsemd (accuracy)')
plt.show()
'''
