import pathlib
import csv
import numpy as np
import sklearn
import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier

def load_data(path, as_gray=False):
    im = Image.open(path)
    if as_gray:
        im = im.convert('L')
    ar = np.asarray(im).astype(np.float32) / 255. # pixel range is 0~1
    return ar

def load_train_dataset(root_dir, flat=False, as_gray=False):
    dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    _X = []
    _y = []
    for d in dirs:
        label = d.name
        for f in d.glob('*.png'):
            x = load_data(f, as_gray=as_gray)
            if flat:
                x = x.flatten()
            _X.append(x)
            _y.append(label)
    return np.array(_X), np.array(_y)

def load_test_dataset(root_dir, flat=False, as_gray=False):
    files = sorted(root_dir.glob('*.png'))
    filenames = []
    _X = []
    for f in files:
        x = load_data(f, as_gray=as_gray)
        if flat:
            x = x.flatten()
        _X.append(x)
        filenames.append(f.name)
    return np.array(_X), filenames

def save_answer(y, filenames, path):
    with path.open('w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        for f, yy in zip(filenames, y):
            writer.writerow([f, yy])

class FeatureExtractor:
    def extract(self, X):
        ################################
        # Code your feature extractor! #
        ################################
        return X

def main():
    train_dir = pathlib.Path('./train')
    test_dir = pathlib.Path('./test')
    out_path = pathlib.Path('./answer.csv')
    X, y = load_train_dataset(train_dir, flat=True, as_gray=True)

    # Build Your Solution!!
    rng = np.random.RandomState(19930213) # K.A
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rng)
    score_val = []
    for idx_tra, idx_val in tqdm.tqdm(kf.split(X, y), total=kf.get_n_splits()):
        X_tra = X[idx_tra]
        y_tra = y[idx_tra]
        X_val = X[idx_val]
        y_val = y[idx_val]
        # Train your model!
        fe = FeatureExtractor()
        model = RidgeClassifier(random_state=rng)
        model.fit(fe.extract(X_tra), y_tra)
        # Validation for model selection
        pred_val = model.predict(fe.extract(X_val))
        score_val.append(sklearn.metrics.accuracy_score(pred_val, y_val))

    # print model performance
    for i, s in enumerate(score_val):
        print('Accuracy (Fold {:02d}): {:04f}'.format(i, s))
    print('Accuracy (mean): {:04f}'.format(np.mean(s)))

    # training for test
    fe = FeatureExtractor()
    model = RidgeClassifier(random_state=rng)
    model.fit(fe.extract(X), y)

    # Test your model!!
    tX, f = load_test_dataset(test_dir, flat=True, as_gray=True)
    pred_test = model.predict(fe.extract(tX))

    # save your answer as csv file!
    save_answer(pred_test, f, out_path)

if __name__ == '__main__':
    main()
