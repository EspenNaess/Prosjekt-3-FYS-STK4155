from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from LastData import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

(X_train, X_test, y_train, y_test), data = featureExtraction(data, False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def plotAccuracyByPenalty(low, high, kernel):
    '''
        plotAccuracyByPenalty:
            low, high: intervall for straffeparametrar ein ønskjer å testa for
            kernel: kernelfunksjon
            
    '''
    cs = [2**i for i in range(low, high)]
    scores = np.empty(len(cs))
    for i in range(len(cs)):
        poly_kernel_svm_clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel=kernel, degree=3, coef0=0.01, C=cs[i]))
            ])

        poly_kernel_svm_clf.fit(X_train, y_train)
        y=poly_kernel_svm_clf.predict(X_test)
        scores[i]=accuracy_score(y_test, y)
        print("Ferdig med {} av {} iterasjonar".format(i+1, len(cs)))

    plt.legend()
    plt.title("Grannsemd som funksjon av log(C) med kernelfunksjon {}".format(kernel))
    plt.xscale('log')
    plt.xlabel('Straffeparameter C')
    plt.ylabel('Grannsemd (accuracy)')
    plt.plot(cs, scores)
    plt.show()

#plotAccuracyByPenalty(-5, 8, "linear")
#plotAccuracyByPenalty(-5, 12, "poly")
#plotAccuracyByPenalty(-5, 8, "sigmoid")
#plotAccuracyByPenalty(-5, 8, "rbf")

# testkøyring

'''
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=2**(2)))
    ])

poly_kernel_svm_clf.fit(X_train, y_train)
y=poly_kernel_svm_clf.predict(X_test)
scoreres=accuracy_score(y_test, y)
print("Accuracy SVM: {}:".format(scoreres))
'''
