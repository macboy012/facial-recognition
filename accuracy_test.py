import utils
import os
import cv2
import sys
from collections import defaultdict


TOLERANCE = 0.35

def uuuh_stats(vals):
    if len(vals) == 0:
        return None
    return sum(vals)/len(vals), min(vals)

def run(directory):
    match_data = utils.read_match_data()
    pnames, faces = utils.get_names_faces_lists(match_data)

    person_scores = defaultdict(lambda: {"tp":0,"tnn":0,"tn":0,"fp":0,"fn":0, 'fn_vals':[], 'tn_vals':[]})


    for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        if os.path.isdir(fullpath) and dir_name != "__pycache__":
            names = os.listdir(fullpath)
            if "name.txt" not in names:
                continue
            with open(os.path.join(fullpath, "name.txt"), "rb") as f:
                truth_name = f.read().strip()
            for img_name in names:
                if img_name == 'name.txt':
                    continue
                if img_name.endswith(".png") or img_name.endswith(".jpg"):
                    encoding = utils.load_and_cache_encoding(fullpath, img_name[:-4], jitters=10)
                else:
                    continue

                if encoding is None:
                    #print 'no face found', dir_name, img_name
                    person_scores[truth_name]['tnn'] += 1
                    continue

                distances = utils.get_face_distances_with_encoding(encoding, faces)
                min_score = 1
                min_name = ''
                for name, score in zip(pnames, distances):
                    if score < min_score:
                        min_name = name
                        min_score = score

                if min_score > TOLERANCE:
                    if truth_name == min_name:
                        person_scores[truth_name]['fn_vals'].append(min_score)
                        person_scores[truth_name]['fn'] += 1
                    else:
                        person_scores[truth_name]['tn_vals'].append(min_score)
                        person_scores[truth_name]['tn'] += 1
                else:
                    if truth_name == min_name:
                        person_scores[truth_name]['tp'] += 1
                    else:
                        if truth_name != 'nevermatch':
                            print "PANIC, FALSE POSITVE - file: %s, true: %s, pred: %s" % (os.path.join(fullpath, img_name), truth_name, min_name)
                            person_scores[truth_name]['fp'] += 1

    #print person_scores
    for name, scores in person_scores.items():
        #print "-------------------------"
        #print name
        sumv = sum([scores['tp'], scores['tn'], scores['fn'], scores['tnn']])
        print name, 1 if sumv == 0 else float(scores['tp'])/sumv
        #print 'tn', uuuh_stats(scores['tn_vals'])
        #print 'fn', uuuh_stats(scores['fn_vals'])
        #print 'avg tn', sum(scores['tn_vals'])/len(scores['tn_vals'])
        #print 'avg fn', sum(scores['fn_vals'])/len(scores['fn_vals'])
        #print utils.format_graph(scores)
        #print scores

"""
tp = score good and match good
tn = score bad and match bad - meh
fp = score good match bad
fn = score bad match good
"""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'need a directory'
    else:
        run(sys.argv[1])
