from __future__ import division
import utils
import os
import cv2
import sys
from collections import defaultdict, Counter


TOLERANCE = 0.35

def uuuh_stats(vals):
    if len(vals) == 0:
        return None
    return sum(vals)/len(vals), min(vals)


model_storage = utils.load_model("modelv2_testing.pkl")
tree_model = utils.TreeModel(model_storage)

def run_vote_with_distance(data, distance):
    data = [d[0] for d in data if d[1] <= distance]
    if len(data) == 0:
        return None, None

    vcount = len(data)
    counter = Counter(data)
    # Too many people found
    if len(counter) > 2:
        return None, None
    sorted_vote = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    #print sorted_vote, vcount//2
    vote = sorted_vote[0][0]
    # If not > half votes and vote count 3 or more.
    if counter[vote] <= vcount//2 or counter[vote] < 3:
        return None, None
    return vote, counter

def process_encodings(encodings, name, data_blob, dir_name):
    just_encodings = [x[0] for x in encodings]
    file_names = [x[1] for x in encodings]
    img_preds = tree_model.get_match_info(just_encodings)
    for preds in img_preds:
        name_dist = []
        for pred in preds:
            # Try and exclude all the photos in the photoset that is under test currently.
            if pred['filepath'].split("/")[-1] in file_names:
                continue
            if pred['distance'] == 0.0:
                continue
            name_dist.append((pred['name'], pred['distance']))
        for dist in [x/100.0 for x in range(20, 33, 1)]:
            voted, counter = run_vote_with_distance(name_dist, dist)
            if dist not in data_blob[name]:
                data_blob[name][dist] = {'hit': 0, 'false': 0, 'attempt': 0}
            peep = data_blob[name][dist]
            peep['attempt'] += 1
            if voted == name:
                peep['hit'] += 1
            elif voted is not None:
                """
                print "dir: %s, voted: %s, name: %s" % (dir_name, voted, name)
                print 'vote count: %s' % len(preds)
                print counter

                for i, x in enumerate(sorted(preds, key=lambda x: x['distance'])):
                    if dist < x['distance']:
                        break
                    print x['distance'], x['name'], x['filepath']
                    img = cv2.imread(x['filepath'], 1)
                    cv2.imshow('image', img)
                    cv2.waitKey(200)
                    if i % 10 == 0 and i != 0:
                        x = raw_input()
                        if x.strip() != "":
                            break
                raw_input()
                """
                peep['false'] += 1

def run(directory):
    #person_scores = defaultdict(lambda: {"tp":0,"tnn":0,"tn":0,"fp":0,"fn":0,"nn":0,"opps":0,"reca":0})
    person_scores = defaultdict(lambda: {})

    dirnames = os.listdir(directory)
    #dirnames = dirnames[:1000]
    for i, dir_name in enumerate(dirnames):
        print "%s/%s" % (i+1, len(dirnames))
        fullpath = os.path.join(directory, dir_name)
        if os.path.isdir(fullpath) and dir_name != "__pycache__":
            names = os.listdir(fullpath)
            if "name.txt" not in names:
                continue
            with open(os.path.join(fullpath, "name.txt"), "rb") as f:
                truth_name = f.read().strip()
            if truth_name == 'nevermatch':
                continue

            encodings = []
            for img_name in names:
                if img_name == 'name.txt':
                    continue
                if img_name.endswith(".png") or img_name.endswith(".jpg"):
                    encoding = utils.load_and_cache_encoding(fullpath, img_name[:-4], jitters=10)
                else:
                    continue

                if encoding is None:
                    #print 'no face found', dir_name, img_name
                    #person_scores[truth_name]['tnn'] += 1
                    continue

                encodings.append((encoding, img_name))
            if len(encodings) == 0:
                continue

            try:
                process_encodings(encodings, truth_name, person_scores, dir_name)
            except ValueError:
                print dir_name
                print encodings
                raise

            """
                person_scores[truth_name]['opps'] += 1

                preds = tree_model.get_predictions([encoding])
                pred = preds[0]
                #distances = utils.get_face_distances_with_encoding(encoding, faces)
                if pred == truth_name:
                    person_scores[pred]['tp'] += 1
                    person_scores[truth_name]['reca'] += 1
                elif pred is None:
                    person_scores[truth_name]['nn'] += 1
                else:
                    person_scores[pred]['fp'] += 1
                """

    #print person_scores
    longest = len(sorted(person_scores.keys(), key=len, reverse=True)[0])
    score_bins = {x:{'hit':0, 'false':0, 'attempt':0} for x in person_scores.values()[0].keys()}

    for scores in person_scores.values():
        for dist, results in scores.items():
            score_bins[dist]['hit'] += results['hit']
            score_bins[dist]['attempt'] += results['attempt']
            score_bins[dist]['false'] += results['false']
    print score_bins
    print "dist,hit,false,attempt"
    for dist, scores in sorted(score_bins.items()):
        print "%s, %s, %s, %s" % (dist, scores['hit'], scores['false'], scores['attempt'])
    return

    for name, scores in person_scores.items():
        if scores.values()[0]['attempt'] < 50:
            continue
        print name, " "* (longest-len(name)),
        for dist, results in sorted(scores.items()):
            print "%s: %.2f - %.2f   " % (dist, round(results['hit']/results['attempt'], 2), round(results['false']/results['attempt'], 2)),
        print '  %s' % results['attempt']

        #if scores['opps'] < 50:
            #continue
        #print "-------------------------"
        #print name
        #sumv = sum([scores['tp'], scores['tn'], scores['fn'], scores['tnn']])
        #print name, 1 if sumv == 0 else float(scores['tp'])/sumv
        #print 'tn', uuuh_stats(scores['tn_vals'])
        #print 'fn', uuuh_stats(scores['fn_vals'])
        #print 'avg tn', sum(scores['tn_vals'])/len(scores['tn_vals'])
        #print 'avg fn', sum(scores['fn_vals'])/len(scores['fn_vals'])
        #print utils.format_graph(scores)
        #print scores
        #print 'tp:', scores["tp"]
        #print 'fp:', scores["fp"]
        #print 'nn:', scores["nn"]
        #print name, "rr:", scores['reca']/float(scores['opps'])

"""
tp = truth == prediction
fp = truth != prediction
"""
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
