import os
import shlex
import argparse
from tqdm import tqdm
import dask.array as da

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap
from dask.diagnostics import ProgressBar

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--n_clusters', default=100, type=int, 
                        help='number of clusters for dictionary')
    parser.add_argument('--max_descriptors', default=500000, type=int, 
                        help='maximum number of descriptors for dictionary')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_

def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors, clusters, k=1)
    
    T, K = len(descriptors), len(clusters)
    assignment = np.zeros((T, K), dtype=np.float32)
    
    for i, match in enumerate(matches):
        assignment[i, match[0].trainIdx] = 1  # Assig

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

        if desc.size == 0:  # Skip empty descriptor sets
            continue

        a = assignments(desc, mus)
        
        T,D = desc.shape
        vlad_enc = np.zeros((K, D), dtype=np.float32)

        for k in range(mus.shape[0]):
            indices = np.where(a[:, k] == 1)[0]
            if len(indices) > 0:
                residuals = desc[indices] - mus[k]
                if gmp:
                    vlad_enc[k] = np.sum(residuals / (np.abs(residuals) + gamma), axis=0)
                else:
                    vlad_enc[k] = residuals.sum(axis=0)

        vlad_enc = vlad_enc.flatten()

   
        # c) power normalization
        if powernorm:
            vlad_enc = np.sign(vlad_enc) * np.sqrt(np.abs(vlad_enc))

        # l2 normalization
        vlad_enc /= np.linalg.norm(vlad_enc)

        encodings.append(vlad_enc)

    return encodings

def esvm(encs_test, encs_train, C=1000):
    """
    Compute E-SVMs using Dask for parallel processing.

    Parameters:
        encs_test: NxD matrix (test encodings)
        encs_train: MxD matrix (train encodings)
        C: Regularization parameter for LinearSVC
    
    Returns:
        new_encs: NxD matrix of new encodings
    """

    # Convert encs_test and encs_train to NumPy arrays if they are lists
    encs_test = np.array(encs_test)
    encs_train = np.array(encs_train)

    # Dask's lazy wrapper for encs_test
    encs_test_dask = da.from_array(encs_test, chunks=(100, encs_test.shape[1]))

    def train_svm_for_encoding(test_encoding):
        """
        Train an individual SVM for the given test encoding.
        """
        X = np.vstack((encs_train, test_encoding[np.newaxis, :]))  # Shape (M+1, D)
        y = np.hstack((np.full(encs_train.shape[0], -1), 1))  # Negatives and one positive

        clf = LinearSVC(C=C, class_weight="balanced", max_iter=1000, tol=1e-3)
        clf.fit(X, y)

        normalized_weights = normalize(clf.coef_, norm="l2")
        
        return normalized_weights.flatten()

    with ProgressBar():  # Wrap the computation with a progress bar bcz it doesnt come with it
        new_encs = encs_test_dask.map_blocks(
            lambda block: np.array([train_svm_for_encoding(row) for row in block]),
            dtype=encs_test.dtype
        ).compute()

    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    encs = normalize(encs, norm='l2')
    similarity = np.dot(encs, encs.T)
    dists = 1 - similarity
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        descriptors = loadRandomDescriptors(
            files=files_train,
            max_descriptors=args.max_descriptors
        )
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # cluster centers
        print('> compute dictionary')
        mus = dictionary(
            descriptors=descriptors,
            n_clusters=args.n_clusters
        )
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # Extract descriptors for test files
        print('> Loading descriptors for test files')
        descriptors_test = [gzip.open(f, 'rb').read() for f in files_test]
        descriptors_test = [cPickle.loads(desc, encoding='latin1') for desc in descriptors_test if len(desc) > 0]
        
        print('> Computing VLAD encodings for test files')
        enc_test = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)

        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        
        print('> Loading descriptors for train files')
        descriptors_train = [gzip.open(f, 'rb').read() for f in files_train]
        descriptors_train = [cPickle.loads(desc, encoding='latin1') for desc in descriptors_train if len(desc) > 0]
        
        print('> Computing VLAD encodings for train files')
        enc_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma)

        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    enc_test_esvm = esvm(enc_test, enc_train, C=1000)

    # eval
    evaluate(enc_test_esvm, labels_test)
    print('> evaluate')
