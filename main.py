import numpy as np
import random
import os
import librosa
import librosa.display

from sklearn.mixture import GaussianMixture

from itertools import product
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from typing import List, Tuple, Dict

import warnings
warnings.filterwarnings('ignore')


class Audio:
    def __init__(self, dir_path:str, cv:int, sampling_rate:int, mfcc_num:int)->None:
        #데이터 경로, sampling rate, mfcc 개수 정의 및 시드 고정
        self.dir_path = dir_path
        self.cv = cv
        self.sampling_rate = sampling_rate
        self.mfcc_num = mfcc_num
        self.seed = 42

        #데이터셋 정의
        self.dataset_x, self.dataset_y = [], []
        
        #음성 파일들 불러온 후 자르기, mfcc 함수 사용
        for index, file in enumerate(os.listdir(dir_path)):
            data, _ = librosa.load(f"{self.dir_path}/{file}",sr=self.sampling_rate)

            data = data[:self.sampling_rate*10]

            for value in self.toMfcc(data):
                self.dataset_x.append(value)
                self.dataset_y.append(index)
            
        self.dataset_x, self.dataset_y = self.shuffle()

        
        # self.dataset_y = np.array(list(self.dataset_y))
        # self.dataset_x = np.array(list(self.dataset_x))

        return None

    def shuffle(self)->Tuple:
        dataset = list(zip(self.dataset_x, self.dataset_y))
        random.shuffle(dataset)
        X, y = zip(*dataset)

        return np.array(list(X)), np.array(list(y))

    def toMfcc(self, data)->list:
        #mfcc 통과
        data_mfcc = librosa.feature.mfcc(y=data, sr=self.sampling_rate, n_mfcc=self.mfcc_num, hop_length=int(self.sampling_rate/100))
        return data_mfcc.T

    def gmmModel(self, params=None)->list:
        class_num = 10

        if params == None:
            estimator = GaussianMixture(n_components=class_num, covariance_type='tied', max_iter=10, reg_covar=1e-2, random_state=self.seed)
        else:
            estimator = GaussianMixture(n_components=class_num, max_iter=10, reg_covar=1e-2, random_state=self.seed, **params)

        # split 개수, 셔플 여부 및 seed 설정
        kf = KFold(n_splits = self.cv, shuffle = False)

        acc = 0
        i = 1
        # kfold로 데이터셋 스플릿
        for train_index, test_index in kf.split(self.dataset_x):
            X_train, X_test = self.dataset_x[train_index], self.dataset_x[test_index]
            y_train, y_test = self.dataset_y[train_index], self.dataset_y[test_index]

            estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(class_num)])
            estimator.fit(X_train)
            
            #혼동 출력 및 정확도 계산
            conf, accuracy =  self.scoreMatrix(estimator.predict(X_test), y_test)
            print(f"cv : {i}\n{conf}")
            i += 1
            acc += accuracy

        print(f"accuracy : {acc/self.cv}")

        return None


    def svcModel(self, params=None)->list:
        #svc 모델, 파라미터 input이 없으면 기본으로 돌림

        if params == None:
            model = SVC(C=1.0, kernel='rbf', random_state=self.seed)
        else:
            model = SVC(random_state=self.seed, **params)

        # split 개수, 셔플 여부 및 seed 설정
        kf = KFold(n_splits = self.cv, shuffle = False)

        acc = 0
        i = 1
        # kfold로 데이터셋 스플릿
        for train_index, test_index in kf.split(self.dataset_x):
            X_train, X_test = self.dataset_x[train_index], self.dataset_x[test_index]
            y_train, y_test = self.dataset_y[train_index], self.dataset_y[test_index]

            model.fit(X_train, y_train)
            conf, accuracy = self.scoreMatrix(list(model.predict(X_test)), y_test)
            print(f"cv : {i}\n{conf}")
            i += 1
            acc += accuracy
        print(f"accuracy : {acc/self.cv}")

        return None

    def scoreMatrix(self, predict:list, actual:list)->Tuple[list,float]:
        #스코어 매트릭스 계산
        conf_mat = np.zeros((10,10))
        for i in range(len(predict)): conf_mat[predict[i]][actual[i]] +=1
        # print("confuse matrix")
        # print(conf_mat)
        no_correct = 0
        for i in range(10): no_correct += conf_mat[i][i]
        accuracy = no_correct/len(predict)
        # print("\n\n accuracy")
        # print(no_correct/len(predict))
        return conf_mat, accuracy

    def scoring(self, predict:list, actual:list)->int:
        #acc만 리턴하는 함수
        conf_mat = np.zeros((10,10))
        for i in range(len(predict)): conf_mat[predict[i]][actual[i]] +=1   
        no_correct = 0
        for i in range(10): no_correct += conf_mat[i][i]
        accuracy = no_correct/len(predict)
        return accuracy

    def tuning(self, model_type:str, params)->Tuple[float, dict]:
        #하이퍼파리마터 튜닝(그리드 서치)

        #acc, params 초기화
        best_acc = 0
        best_parmas = None

        #params 순열 생성
        grid_list = [dict(zip(params, v)) for v in product(*params.values())]
        
        #gmm일 경우
        if model_type == 'gmm':
            class_num = 10

            kf = KFold(n_splits = self.cv, shuffle = False)
            #그리드 서치
            for grid in grid_list:
                estimator = GaussianMixture(n_components=class_num, max_iter=10, random_state=self.seed, reg_covar=1e-2, **grid)

                acc = 0
                for train_index, test_index in kf.split(self.dataset_x):
                    X_train, X_test = self.dataset_x[train_index], self.dataset_x[test_index]
                    y_train, y_test = self.dataset_y[train_index], self.dataset_y[test_index]

                    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(class_num)])
                    estimator.fit(X_train)
                    acc += self.scoring(estimator.predict(X_test), y_test)
                
                acc /= self.cv

                print(f"{grid} : {acc}")

                #평가
                if best_acc <= acc:
                    best_acc = acc
                    best_parmas = grid

            return best_acc, best_parmas
        
        #svc일 경우
        elif model_type == 'svc':
            #그리드 서치
            for grid in grid_list:
                model = SVC(random_state=self.seed, **grid)

                kf = KFold(n_splits = self.cv, shuffle = False)

                acc = 0
                for train_index, test_index in kf.split(self.dataset_x):
                    X_train, X_test = self.dataset_x[train_index], self.dataset_x[test_index]
                    y_train, y_test = self.dataset_y[train_index], self.dataset_y[test_index]


                    model.fit(X_train, y_train)
                    acc += self.scoring(list(model.predict(X_test)), y_test)

                acc /= self.cv

                print(f"{grid} : {acc}")

                if best_acc <= acc:
                    best_acc = acc
                    best_parmas = grid
                
            return best_acc, best_parmas
        
        else: return None


if __name__ == '__main__':
    random.seed(42)
    #mfcc 그리드서치
    num_mfcc=80

    print(f"mfcc : {num_mfcc}")
    
    #class 객체 정의
    audio_obj = Audio('./data', 4, 16000, num_mfcc)
    
    #모델별 하이퍼파라미터할 파라미터 정의
    svc_parameters = {'kernel': ('linear','rbf','poly','sigmoid'), 'C':[4,8,16]}
    gmm_parmas = {'covariance_type':('diag','tied'), 'tol':(1e-2,1e-3,1e-4)}

    #svc 하이퍼파라미터 튜닝
    print(f"mfcc : {num_mfcc} ) svc hyper parameter tuning ----")
    svc_best_acc, svc_best_parmas = audio_obj.tuning('svc', svc_parameters)
    print(f"best_acc : {svc_best_acc},\nbest_params : {svc_best_parmas}", end='\n\n')

    print(f"mfcc : {num_mfcc} ) svc fit ----")
    svc_model_pred = audio_obj.svcModel(svc_best_parmas)
    # svc_model_pred = audio_obj.svcModel()

    print(f"mfcc : {num_mfcc} ) gmm hyper parameter tuning ----")
    gmm_best_acc, gmm_best_parmas = audio_obj.tuning('gmm', gmm_parmas)
    print(f"best_acc : {gmm_best_acc},\nbest_params : {gmm_best_parmas}", end='\n\n')

    print(f"mfcc : {num_mfcc} ) gmm fit ----")
    gmm_model_pred = audio_obj.gmmModel(gmm_best_parmas)
    # gmm_model_pred = audio_obj.gmmModel()



    # for num_mfcc in [40,80]:
    #     print(f"mfcc : {num_mfcc}")

    #     #class 객체 정의
    #     audio_obj = Audio('./data', 4, 16000, num_mfcc)
        
    #     #모델별 하이퍼파라미터할 파라미터 정의
    #     svc_parameters = {'kernel': ('linear','rbf','poly','sigmoid'), 'C':[4,8,16]}
    #     gmm_parmas = {'covariance_type':('diag','tied'), 'tol':(1e-2,1e-3,1e-4)}

    #     #svc 하이퍼파라미터 튜닝
    #     print(f"mfcc : {num_mfcc} ) svc hyper parameter tuning ----")
    #     svc_best_acc, svc_best_parmas = audio_obj.tuning('svc', svc_parameters)
    #     print(f"best_acc : {svc_best_acc},\nbest_params : {svc_best_parmas}", end='\n\n')


    #     print(f"mfcc : {num_mfcc} ) svc fit ----")
    #     svc_model_pred = audio_obj.svcModel(svc_best_parmas)
    #     # svc_model_pred = audio_obj.svcModel()

    #     if num_mfcc < 120:
    #         print(f"mfcc : {num_mfcc} ) gmm hyper parameter tuning ----")
    #         gmm_best_acc, gmm_best_parmas = audio_obj.tuning('gmm', gmm_parmas)
    #         print(f"best_acc : {gmm_best_acc},\nbest_params : {gmm_best_parmas}", end='\n\n')

    #         print(f"mfcc : {num_mfcc} ) gmm fit ----")
    #         gmm_model_pred = audio_obj.gmmModel(gmm_best_parmas)
    #         # gmm_model_pred = audio_obj.gmmModel()
