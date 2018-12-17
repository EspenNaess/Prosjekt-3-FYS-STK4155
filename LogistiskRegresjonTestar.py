from sklearn.linear_model import LogisticRegression
from LastData import *
from sklearn.metrics import accuracy_score
import warnings

(X_train, X_test, y_train, y_test), data = featureExtraction(data, False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# eit vanleg testdøme med optimale parametrar
'''
mlp = LogisticRegression(C=5*10**(-3), penalty='l1')
p=0.1
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp.fit(X_train[:8],y_train[:8])

y=mlp.predict(X_test)
scoreres=accuracy_score(y_test, y)

mlp.fit(X_train, y_train)
y=mlp.predict(X_test)
print(accuracy_score(y_test, y))
'''

def graphAccuracyByDataPercentage(percentages):
    '''
        graphAccuracyByDataPercentage:
            Lagar eit plott over grannsemd som ein funksjon av prosentdel av treningsmengda som er nytta

            param:
                percentages: liste over dataprosentar som skal plottast for
    '''
    scores = np.empty(len(percentages))
    for i in range(len(percentages)):
        mlp = LogisticRegression(C=5*10**(-3), penalty='l1')

        p=percentages[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train[:int(p*len(X_train))],y_train[:int(p*len(y_train))])

        y=mlp.predict(X_test)
        scoreres=accuracy_score(y_test, y)

        mlp.fit(X_train, y_train)
        y=mlp.predict(X_test)
        scores[i]=accuracy_score(y_test, y)
        print("Ferdig med {} av {} iterasjonar".format(i+1, len(percentages)))

    plt.legend()
    plt.title("Grannsemd som funksjon av straffeparameter C")
    #plt.xscale('log')
    plt.xlabel('Prosent av datamengda')
    plt.ylabel('Grannsemd (accuracy)')
    plt.plot(percentages, scores)
    plt.show()

def plotDecisionBoundary(): # lagar eit plott over avgjerdsgrensa med dei mest optimale parametrane som er funne tidlegare
    mlp = LogisticRegression(C=5*10**(-3), penalty='l1')

    p=1 # heile datamengda nytta
    print(int(p*len(X_train)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mlp.fit(X_train[:int(p*len(X_train))],y_train[:int(p*len(y_train))])

    y=mlp.predict(X_test)
    scoreres=accuracy_score(y_test, y)
    print("Accuracy LR: {}:".format(scoreres))

    xx, yy = np.mgrid[-1000:1500, -1000:2000]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))

    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    ax.scatter(data[100*(nCat-1)*23:,0], data[100*(nCat-1)*23:,1], color='red', label='Anfall', s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    ax.scatter(data[:100*(nCat-1)*23,0], data[:100*(nCat-1)*23,1], color='blue', label="Inkje anfall", alpha=0.5, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlabel="$X_1$", ylabel="$X_2$")

    plt.title("Avgjerdsgrense for logistisk regresjon")
    plt.legend()
    plt.show()

def plotAccuracyByPenalty(low, high):
    '''
        plotAccuracyByPenalty:
            Lagar eit plott over grannsemd som ein funksjon av straffeparameterverdiar

            param:
                low, high: intervall for straffeparametrar ein ønskjer å testa for
                kernel: kernelfunksjon

    '''
    cs = np.logspace(low, high)
    scores = np.empty(len(cs))
    for i in range(len(cs)):
        mlp = LogisticRegression(C=cs[i], penalty='l1')

        p=1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mlp.fit(X_train[:int(p*len(X_train))],y_train[:int(p*len(y_train))])

        y=mlp.predict(X_test)
        scoreres=accuracy_score(y_test, y)

        mlp.fit(X_train, y_train)
        y=mlp.predict(X_test)
        scores[i]=accuracy_score(y_test, y)
        print("Ferdig med {} av {} iterasjonar".format(i+1, len(cs)))

    plt.legend()
    plt.title("Grannsemd som funksjon av straffeparameter C")
    plt.xscale('log')
    plt.xlabel('Straffeparameter C')
    plt.ylabel('Grannsemd (accuracy)')
    plt.plot(cs, scores)
    plt.show()

#plotAccuracyByPenalty(-5, 3)
#plotDecisionBoundary()
#graphAccuracyByDataPercentage(np.arange(1,101,1)*0.001)
