import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interp
from lbp_module.texture import base_lbp, improved_lbp, hamming_lbp, completed_lbp, extended_lbp
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pygame import mixer

ALGORITHM = {
    'base_lbp': base_lbp,
    'improved_lbp': improved_lbp,
    'extended_lbp': extended_lbp,
    'completed_lbp': completed_lbp,
    'hamming_lbp': hamming_lbp
}

VARIANTS = {
    'base_lbp': 'ORIGINAL LBP',
    'improved_lbp': 'IMPROVED LBP',
    'extended_lbp': 'EXTENDE LBP',
    'completed_lbp': 'COMPLETED LPB',
    'hamming_lbp': 'HAMMING LBP'
}

def main():
    DATASET = 'DATASET/Breast Cancer/BreakHist_Dataset/40X'
    VARIANT = 'extended_lbp'
    METHOD = 'ror' # default, uniform, nri_uniform, ror, var
    P, R = 16, 2

    print('Resultados: P = {}, R = {}'.format(P, R))
    x_data = []
    y_data = []

    print('Computando recursos...')
    for index_binary, _class in enumerate(os.listdir(DATASET)):
        print('Subtipo {}...'.format(_class))
        path_class = os.path.join(DATASET, _class) # Malign and Benign

        for _subclass in os.listdir(path_class):
            path_img_subclass = os.path.join(path_class, _subclass)

            print('Classe {}: {}'.format(index_binary, _subclass))
            
            for name_img in tqdm(os.listdir(path_img_subclass)):
                path_name_img = os.path.join(path_img_subclass, name_img)
                img = cv2.imread(path_name_img, cv2.IMREAD_GRAYSCALE)
                
                if img.shape != (460, 700):
                    img = cv2.resize(img, (460, 700))
                
                feature = ALGORITHM[VARIANT](img, P=P, R=R, block=(1, 1), method=METHOD)

                x_data.append(list(feature))
                y_data.append(index_binary)

    x_data = np.asarray(x_data)#normalize(np.asarray(x_data), norm='l1') # #
    y_data = np.asarray(y_data)

    print('Tamanho do vetor de recursos: ', x_data[0].shape)
    
    n_cv = 5

    svm = SVC(probability=True)
    mlp = MLPClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()
    
    svm_parameters = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 10, 100],
        'gamma': [0.0001, 0.00001, 0.000001]
    }
    mlp_parameters = {
        'hidden_layer_sizes': [(5,), (10,), (20,), (10, 10)],
        'solver': ['adam', 'sgd'],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'max_iter': [50, 100, 200]
    }
    dt_parameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 10, 50],
    }
    rf_parameters = {
        'n_estimators': [5, 11, 51, 101],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 50, 100, 200],
    }
    knn_parameters = {
        'n_neighbors': [1, 5, 9],
        'weights' : ['uniform', 'distance'],
        'algorithm': ['kd_tree', 'ball_tree'],
        'p': [1, 2] # Manhatan and Euclidian distance, respectivity
    }

    classifiers = [['SVM', svm, svm_parameters], ['MLP', mlp, mlp_parameters], \
                ['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
                ['K-Nearest Neighbor', knn, knn_parameters]]
    
    classifiers = [['MLP', mlp, mlp_parameters], ['Decision Trees', dt, dt_parameters], ['Random Forest', rf, rf_parameters], \
                ['K-Nearest Neighbor', knn, knn_parameters]]
    
    # METRICAS A SEREM ANALISADAS
    recall = []
    precision = []
    f1score = []
    accuracy = []

    for _id, clf, parameters in classifiers:
        np.random.seed(10)
        cv = StratifiedKFold(n_splits=n_cv)
        print(35 * ' * ')
        print('Classificando com {}...'.format(_id))
        
        # CROSS-VALIDATION
        aucs, tprs, results = [], [], []
        mean_fpr = np.linspace(0, 1, 100)
        
        clf_grid_search = GridSearchCV(clf, param_grid=parameters, scoring='accuracy', cv=5)

        print('Iniciando GridSearch...')
        results_grid_search = clf_grid_search.fit(X=x_data, y=y_data)

        print('Melhor Parametro: {}'.format(results_grid_search.best_params_))
        print('Melhor F1Score: '.format(np.round(results_grid_search.best_score_, 2)))
        print(35 * '- ')
        print()

        np.random.seed(10)

        for i, (train, test) in enumerate(cv.split(x_data, y_data)):

            best_clf = clf_grid_search.best_estimator_
            
            probas_ = best_clf.fit(x_data[train], y_data[train]).predict_proba(x_data[test])
            
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.5,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            
            # COMPUTANDO METRICAS PARA CADA FOLD
            i_recall = recall_score(y_data[test], np.round(probas_[:, 1]))
            i_precision = precision_score(y_data[test], np.round(probas_[:, 1]))
            i_f1score = f1_score(y_data[test], np.round(probas_[:, 1]))
            i_accuracy = accuracy_score(y_data[test], np.round(probas_[:, 1]))
            
            recall.append(i_recall)
            precision.append(i_precision)
            f1score.append(i_f1score)
            accuracy.append(i_accuracy)
        
        #PLOTANDO LINHA DIAGONAL --> y = x
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', \
                label='Chance', alpha=.8)

        # PLOTANDO MEDIA DA CURVA ROC DOS FOLDS
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', \
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), \
                lw=2, alpha=.8)

        # PLOTANDO NUVEM DO DESVIO PADRÃO
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

        mixer.init()
        mixer.music.load('/home/marcelo/Música/y2mate.com - ed_sheeran_perfect_official_music_video_2Vv-BfVoq4g.mp3')
        mixer.music.play()

        # PLOTANDO INFORMACOES BASICA DO GRAFICO
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - {} - {}/{} - P: {}, R:{}'.format(_id, METHOD.upper(), VARIANTS[VARIANT], P, R))
        plt.legend(loc="lower right")
        
        if not os.path.exists(VARIANT):
            os.makedirs(VARIANT)

        plt.savefig('{}/{}_{}_{}_{}_{}.png'.format(VARIANT, METHOD.upper(), VARIANTS[VARIANT], _id.replace(' ', ''), P, R))
        print('{}_{}_{}_{}_{}.png'.format(METHOD.upper(), VARIANTS[VARIANT], _id.replace(' ', ''), P, R))
        
        #plt.show()
        plt.close()

        # PRINTANDO RESULTADOS GERAIS DO MODELO
        results = np.asarray([accuracy, recall, precision, f1score])
        id_results = ['accuracy', 'precision', 'recall', 'f1score']
        
        print('Resultados do Classificador: {}'.format(_id))
        for i, res in enumerate(results):
            results_mean = 100 * np.round(res.mean(), 4)
            results_std = 100 * np.round(res.std(), 4)
            score = id_results[i]
            print('Resultados {}: {}'.format(score, res))
            print('Média: {}%'.format(results_mean))
            print('Desvio Padrão: {}%'.format(results_std))
            print(35 * '- ')

if __name__ == '__main__':

    main()
