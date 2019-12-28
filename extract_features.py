import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
import cv2
from glob import iglob
from lbp_module.texture import base_lbp, improved_lbp, hamming_lbp, completed_lbp, extended_lbp
from lbp_module.classifiers import Classifier


def main():

    DATASET = 'DATASET/Breast Cancer/BreakHist_Dataset/40X'

    VARIANTS = {
        'BLBP': base_lbp,
        'ILBP': improved_lbp,
        'HLBP': hamming_lbp,
        'CLBP': completed_lbp,
        'ELBP': extended_lbp
    }

    # Parametros
    current_variant = 'BLBP'
    METHOD = 'uniform'
    PARAMETER = {
        'R': [1, 2, 3, 4, 5],
        'P': [4, 8, 12, 16, 20, 24]
    }
    P, R = 24, 3
    BLOCK = (5, 5)
    index = 0
    x_data = []
    y_data = []

    t0 = time.time()
    print('Variante atual: ', current_variant)
    print('Metodo Atual: ', METHOD)
    print('Lendo imagens e computando Descritores LBP: ')

    for _class in os.listdir(DATASET):
        path_class = os.path.join(DATASET, _class) # Malign and Benign
        
        for _subclass in os.listdir(path_class):
            path_img_subclass = os.path.join(path_class, _subclass)
            
            print('Classe {}: {}'.format(index, _subclass))
            
            for name_img in tqdm(os.listdir(path_img_subclass)):
                path_name_img = os.path.join(path_img_subclass, name_img)
                img = cv2.imread(path_name_img, cv2.IMREAD_GRAYSCALE)
                
                if img.shape != (460, 700):
                    img = cv2.resize(img, (460, 700))
                
                feature = VARIANTS[current_variant](img, P=P, R=R, method=METHOD, block=BLOCK)
                
                x_data.append(list(feature))
                y_data.append(index)     
        index += 1          

    print('Tempo Gasto: {} min.'.format((time.time() - t0)/60))
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    print(x_data.shape)
    print(y_data.shape)

    print('Descritores LBP computados...')
    print(60 * '#')

    classifiers = [
                    ['svm', Classifier(classifier='SVM', C=1.0, kernel='rbf')],
                    #['mlp', Classifier(classifier='MLP',)],
                    #['dt', Classifier(classifier='DT',)],
                    #['rf', Classifier(classifier='RF',)],
                    ['knn', Classifier(classifier='NN', K=5)]
                  ]

    for name_clf, clf in classifiers:

        print('Iniciando treinamento utilizando ', name_clf)
        results = clf.cross_validation(x_data, y_data, cv=3, scoring=['accuracy', 'f1_micro','recall_micro', 'precision_micro', 'roc_auc'])

        print('Resultados:')
        metrics = ['test_accuracy', 'test_f1_micro', 'test_precision_micro', 'test_recall_micro', 'test_roc_auc']
        df = pd.DataFrame(results)
        print(df[metrics])

        print(60*'#')


if __name__ == '__main__':

    main()
