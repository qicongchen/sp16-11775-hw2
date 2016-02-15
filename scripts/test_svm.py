#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_suffix feat_type feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_suffix -- suffix of feature files, eg: spbof"
        print "feat_type -- type of feature files, dense|sparse"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_suffix = sys.argv[3]
    feat_type = sys.argv[4]
    feat_dim = int(sys.argv[5])
    output_file = sys.argv[6]

    # load the kmeans model
    svm = cPickle.load(open(model_file, "rb"))

    video_ids = []
    # read in labels
    label_file = "list/test"
    fread_label = open(label_file, 'r')
    for line in fread_label.readlines():
        tokens = line.strip().split(' ')
        video_id = tokens[0]
        video_ids.append(video_id)
    fread_label.close()

    # read in features
    features = []
    for video_id in video_ids:
        feat_path = feat_dir + video_id + "." + feat_suffix
        feature = [0]*feat_dim
        if os.path.exists(feat_path) is True:
            if feat_type == 'dense':
                feature = numpy.genfromtxt(feat_path, delimiter=';')
            else:
                line = numpy.genfromtxt(feat_path, delimiter=' ', dtype=str)
                for item in line:
                    if len(item) == 0:
                        continue
                    tokens = item.split(':')
                    key = int(tokens[0])-1
                    value = float(tokens[1])
                    feature[key] = value
        features.append(feature)

    # test svm
    scores = [sample[1] for sample in svm.predict_log_proba(features)]
    # dump result
    fwrite = open(output_file, 'w')
    for score in scores:
        fwrite.write("%s\n" % str(score))
    fwrite.close()
