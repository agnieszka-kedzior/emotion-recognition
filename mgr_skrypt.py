# %% [markdown]
# # Machine learning notebook from master thesis
# # Artificial intelligence methods in recognizing the emotional state of person based on video stream processing

# %%
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb
import numpy as np
import time
import sys
import cv2
import dlib
import os

# %%
sys.argv = [
    'arg0',
    '../../Dane/shape_predictor_68_face_landmarks.dat',
    '../../Dane/train',
    '../../Dane/test'
]

# %% [markdown]
# # functions

# %%
def get_directories(path):
    print(f'Start: get_directories') 
    list = [f.path for f in os.scandir(path) if f.is_dir()]
    return(list)

# %%
def get_data_np(folders, predictor):
    print(f'Start: get_data_np') 
    features = []
    emotions = []

    for folder in folders:
        directory = os.fsdecode(folder)
        emotion_name = os.path.basename(directory)

        for idx, item in enumerate(os.listdir(directory)):
            filename = os.fsdecode(item)

            if filename.endswith(".jpg"):
                path_to_file = os.path.join(directory, filename)
                img = cv2.imread(path_to_file)

                # Target emotion
                emotions.append(emotion_name)

                # Face landmarks with dlib 
                face = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
                landmarks = predictor(image=img, box=face)
                array = np.array([[pt.x, pt.y] for pt in landmarks.parts()])

                # Normalize array
                scaler = preprocessing.MinMaxScaler()
                scaler.fit(array)
                norm = scaler.transform(array)
                
                features.append(norm)
                continue
            else:
                continue

    features = np.asarray(features)
    emotions = np.asarray(emotions)
    
    # Reshape 3 to 2 dim
    x = np.array(features).reshape(len(features), 2*68)
    return x, emotions

# %%
def start_up(start_time):
    print(f'Start: start_up')
    print("--- %s sec ---" % round((time.time() - start_time), 2))
    arg_predict = sys.argv[1]
    arg_train_dir = sys.argv[2]
    arg_test_dir = sys.argv[3]

    print(f'Start: get_detector_predictor')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(arg_predict)

    folders_train = get_directories(arg_train_dir)
    folders_test = get_directories(arg_test_dir)
    x_train, y_train = get_data_np(folders_train, predictor)
    x_test, y_test = get_data_np(folders_test, predictor)

    # preprecessor
    print(f'Start: start_preprocessing')
    lenc = preprocessing.LabelEncoder()
    lenc.fit(np.ravel(y_train.reshape(-1,1)))
    Y_data = lenc.transform(np.ravel(y_train.reshape(-1,1)))
    Y_new = lenc.transform(np.ravel(y_test.reshape(-1,1)))
    #lenc.classes_ = np.append(lenc.classes_, '<unknown>') #add if needed
    print('classes: ',list(lenc.classes_))
    print("--- %s sec ---" % round((time.time() - start_time), 2))
    print('Ready')
    
    return detector, predictor, x_train, Y_data, x_test, Y_new, lenc

# %% [markdown]
# # run machine learning algorytms

# %%
def run_results(start_time, valid_, pred_, lenc):
    print(f'Start: run_results')
    print("--- %s sec ---" % round((time.time() - start_time), 2))

    # postprecessor
    pred_ = pred_.astype(int)
    Y_val = lenc.inverse_transform(np.ravel(valid_.reshape(-1,1)))
    Y_pred = lenc.inverse_transform(np.ravel(pred_.reshape(-1,1)))

    accuracy = metrics.accuracy_score(valid_, pred_)
    print("Accuracy: ", np.round(accuracy, 3))

    mean_squared_error = metrics.mean_squared_error(valid_, pred_)
    print("MSE: ", np.round(mean_squared_error, 3))

    #cf_matrix = metrics.confusion_matrix(Y_val, Y_pred)
    #print("Confusion matix: ")
    #print(cf_matrix)

    print(metrics.classification_report(Y_val, Y_pred))
    return

# %%
def run_supervised_machine_learning(start_time, model, X_data, Y_data, X_new):
    print(f'Start: run_supervised_machine_learning ', model)
    print("--- %s sec ---" % round((time.time() - start_time), 2))
    model.fit(X_data, Y_data)
    pred = model.predict(X_new)
    return model, pred

# %%
def run_unsupervised_machine_learning(model, X_data, Y_data, X_new):
    print(f'Start: run_unsupervised_machine_learning ', model)
    model.fit(X_data)
    pred = model.predict(X_new)
    return model, pred

# %%
def run_supervised_method(start_time, X_data, Y_data, X_new, idx):
    print(f'Start: run_supervised_method')
    print("--- %s sec ---" % round((time.time() - start_time), 2))
    clf_array = []
    clf_array.append(svm.SVC())
    clf_array.append(GaussianNB())
    clf_array.append(KNN())
    clf_array.append(tree.DecisionTreeClassifier())
    clf_array.append(LinearRegression())
    clf_array.append(svm.SVR(kernel="rbf", C=1.0, epsilon=0.1))
    clf_array.append(ensemble.RandomForestClassifier(max_depth=7, random_state=0))
    clf_array.append(ensemble.AdaBoostClassifier())
    clf_array.append(ensemble.GradientBoostingClassifier())
    clf_array.append(lgb.LGBMClassifier())
    clf_array.append(tree.DecisionTreeRegressor())
    clf_array.append(MLPClassifier())

    model = clf_array[idx]
    model, pred = run_supervised_machine_learning(start_time, model, X_data, Y_data, X_new)
    return model, pred

# %%
def run_unsupervised_method(X_data, Y_data, X_new, idx):
    print(f'Start: run_unsupervised_method')
    clf_array = []
    clf_array.append(DBSCAN())
    clf_array.append(AgglomerativeClustering())
    clf_array.append(GaussianMixture(n_components=68, random_state=0))

    model = clf_array[idx]
    model, pred = run_unsupervised_machine_learning(model, X_data, Y_data, X_new)
    return model, pred

# %% [markdown]
# # detect and predict from videostream

# %%
def run_video(start_time, detector, predictor, model, lenc):
    print(f'Start: run_video')
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()

        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1 = face.left() # left point
            y1 = face.top()  # top point
            x2 = face.right() # right point
            y2 = face.bottom() # bottom point
            # Draw a rectangle
            #cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

            landmarks = predictor(image=gray,box=face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                #cv2.circle(img=frame, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

            array = np.array([[pt.x, pt.y] for pt in landmarks.parts()])

            # Normalize array
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(array)
            norm = scaler.transform(array)

            # Predict
            k = model.predict(norm.reshape(1,-1))
            k = abs(k.astype(int)) #keep if needed
            em = lenc.inverse_transform(k)
            print(f'Emotion at', round((time.time() - start_time), 2),': ', str(em[0]))
            cv2.putText(frame,str(em[0]), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        cv2.imshow(winname="Emotion recognition", mat=frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):# click Q to close
            break

    vid.release()
    cv2.destroyAllWindows()

# %%
def main():
    start_time = time.time()
    detector, predictor, X_data, Y_data, X_new, Y_new, lenc = start_up(start_time)
    model, pred = run_supervised_method(start_time, X_data, Y_data, X_new, 1)
    run_results(start_time, Y_new, pred, lenc)
    run_video(start_time, detector, predictor, model, lenc)

# %%
if __name__ == '__main__':
    main()


