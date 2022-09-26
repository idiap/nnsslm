"""
evaluation.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import os
import pickle as pickle
import math
from operator import itemgetter

import numpy as np
import scipy.spatial

import apkit

from . import utils
from .doa import loc_to_id

_RESULT_DIR = 'results'
_RESULT_SUFFIX = '.rs'

_GT_DIR = 'data'
_WAV_SUFFIX = '.wav'
_GT_SUFFIX = '.gt.pickle'
#_NEW_GT_SUFFIX = '.gtf.pkl'

_BF_RESULT_DIR = 'sns_gtdoa_results'
_BF_DATA_DIR = 'bf'
_STYPE_SUFFIX = '.stype.pkl'

_ONE_DEGREE = math.pi / 180.0


def save_results(root, name, result):
    """Save prediction results to disk.

    Args:
        root   : path to database root
        name   : name of method
        result : prediction results, list of (id, prediction) tuples.
    """
    path = root + '/' + _RESULT_DIR + '/' + name + _RESULT_SUFFIX
    with open(path, 'w') as f:
        pickle.dump(result, f)


def _norm(a):
    return math.sqrt(np.dot(a, a))


def angular_distance(a, b):
    denom = (_norm(a) * _norm(b))
    if denom < 1e-16:
        return math.pi
    sim = np.dot(a, b) / denom
    if sim > 1.0:
        return 0.0
    else:
        return math.acos(sim)


def azimuth_distance(a, b):
    return angular_distance(a[:2], b[:2])


def cluster_center(data, weight):
    m = np.average(data, axis=0, weights=weight)
    return m / _norm(m)


def load_rng_utt(root, name):
    """Load results and ground truth at utterance level

    Args:
        root   : path to database root
        name   : name of method

    Returns:
        res_gt : list of (id, ground_truth, prediction)
    """
    path = root + '/' + _RESULT_DIR + '/' + name + _RESULT_SUFFIX
    with open(path, 'r') as f:
        lres = pickle.load(f)

    gtpat = root + '/' + _GT_DIR + '/%s' + _GT_SUFFIX
    return [(sid, utils.load_source_location_utt(gtpat % sid), predict)
            for sid, predict in lres]


def load_rng_frame(root, name, vad_suffix, non_sil):
    """Load results and ground truth at frame level

    Args:
        root     : path to database root
        name     : name of method
        vad_suffix : VAD ground truth file suffix.
        non_sil  : non-silent frames only

    Returns:
        res_gt : list of (id, ground_truth, prediction),
                 where id is "SEG_ID:FRAME_ID"
    """
    path = root + '/' + _RESULT_DIR + '/' + name + _RESULT_SUFFIX
    with open(path, 'r') as f:
        lres = pickle.load(f)

    return [('%s:%d' % (sid, fid), gt_locs, predict[fid])
            for sid, predict in lres
            for fid, gt_locs in utils.load_source_location_frame(
                root, sid, vad_suffix,
                utils.LOAD_ACTIVE if non_sil else utils.LOAD_CERTAIN)]


def load_indiv_rng(root, name, sid, vad_suffix):
    path = root + '/' + _RESULT_DIR + '/' + name + _RESULT_SUFFIX
    with open(path, 'r') as f:
        lres = pickle.load(f)

    output = None
    for s, o in lres:
        if s == sid:
            output = o
    assert output is not None

    return output, [
        gt_locs for _, gt_locs in utils.load_source_location_frame(
            root, sid, vad_suffix, utils.LOAD_ALL)
    ]


def match_gt(est, gt, metric, tol):
    gt = np.array(gt)

    match = {}
    if len(est) > 0 and len(gt) > 0:
        # distance
        dis = scipy.spatial.distance.cdist(gt, est, metric=metric)
        dis = [(i, j, dis[i, j]) for i in range(len(gt))
               for j in range(len(est))]
        dis = sorted(dis, key=itemgetter(2))

        # match greedy
        it = iter(dis)
        mest = set()
        while len(match) < min(len(gt), len(est)):
            i, j, d = next(it)
            if d > tol:
                break
            if i not in match and j not in mest:
                match[i] = (j, d)
                mest.add(j)

    return match


def eval_recall_precision(res_gt,
                          etol,
                          verbose=False,
                          metric=azimuth_distance):
    """Evaluate SSL

    Args:
        res_gt  : list of result and ground truth,
                  loaded by load_rng_utt or load_rng_frame
        etol    : admissible error (radian) as a list
        verbose : if True, print result one by one

    Returns:
        nsrc    : number of sources
        recall  : list of recall
        prec    : list of precision
        rmse    : list of root mean squared error of correct detections
        mae     : list of mean absolute error of correct detections
    """
    nsrc = 0
    ncor = np.zeros(len(etol))  # number of correct
    nfal = np.zeros(len(etol))  # number of false alarm
    acse = np.zeros(len(etol))  # accumulation of squared error
    acae = np.zeros(len(etol))  # accumulation of absolute error

    for sid, gt, pred in res_gt:
        if verbose:
            print('%s [%s] [%s]' % \
                    (sid,
                     ', '.join([str(apkit.vec2ae(x)) for x in gt]),
                     ', '.join([str(apkit.vec2ae(x)) for x in pred])))
        nsrc += len(gt)
        match = match_gt(pred, gt, metric, max(etol))
        for i, tol in enumerate(etol):
            nfal[i] += len(pred) - len(match)
            for _, (_, dis) in match.items():
                if dis <= tol:
                    ncor[i] += 1
                    acse[i] += dis * dis
                    acae[i] += dis
                else:
                    nfal[i] += 1
    rmse = np.zeros(len(etol))
    mae = np.zeros(len(etol))
    mask = ncor != 0.0
    rmse[mask] = np.sqrt(acse[mask] / ncor[mask])
    mae[mask] = acae[mask] / ncor[mask]
    return nsrc, ncor / nsrc, ncor / (ncor + nfal), rmse, mae


def print_eval(res_gt):
    """Evaluate SSL at utterance level and print result
    """
    etol = [1.0, 2.0, 5.0, 10.0, 180.1]  # error tolerance
    etolr = [t / 180.0 * math.pi for t in etol]  # as rad

    nsrc, recall, prec, rmse, mae = eval_recall_precision(res_gt, etolr)

    print('nunit=%d' % len(res_gt))
    print('nsrc=%d' % nsrc)
    for i, tol in enumerate(etol):
        print('etol=%.0f recall=%.3f prec=%.3f rmse=%.1f mae=%.1f' % \
                (tol, recall[i], prec[i], rmse[i] / math.pi * 180,
                 mae[i] / math.pi * 180))


def gen_figure(ylabel,
               xlabel,
               titles,
               xdata,
               ydata,
               data_opt='',
               title='',
               yrange=(0., 1.),
               xrange_=(0., 1.)):
    script = []
    script.append('#generated by evaluation.py, copyright: Weipeng He')
    if title is not None and len(title) > 0:
        script.append('set title "%s"' % title)
    script.append('set key noenhanced')
    script.append('set size square')
    script.append('set grid')
    script.append('set tics scale 0')
    script.append('set key left bottom')
    script.append('set style data lines')
    script.append('set ylabel "%s"' % ylabel)
    script.append('set xlabel "%s"' % xlabel)
    if xrange_ is not None:
        script.append('set xrange [%g:%g]' % xrange_)
    if yrange is not None:
        script.append('set yrange [%g:%g]' % yrange)
    script.append('plot ' + ', '.join(
        ['"-" using 1:2 %s lw 2 title "%s"' % (data_opt, n) for n in titles]))

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    if xdata.ndim == 1:
        for yrow in ydata:
            for x, y in zip(xdata, yrow):
                script.append('%f %f' % (x, y))
            script.append('e')
    else:
        assert xdata.ndim == 2
        assert xdata.shape == ydata.shape
        for i in range(len(xdata)):
            xrow = xdata[i]
            yrow = ydata[i]
            for x, y in zip(xrow, yrow):
                script.append('%f %f' % (x, y))
            script.append('e')

    return '\n'.join(script)


def gen_pr_e_figures(l_res_gt, names):
    """Generate precision and recall vs admissible error figures' gnuplot
       script

    Args:
        l_res_gt : list of result gt
        names    : list of method names

    Returns:
        sc_recall : script (as str) of plotting recall
        sc_prec   : script (as str) of plotting precision
    """
    etol = np.arange(1, 11)  # 1 to 10 degrees
    etolr = etol / 180.0 * math.pi  # in radian

    eval_res = [eval_recall_precision(res_gt, etolr) for res_gt in l_res_gt]
    l_recall = [recall for nsrc, recall, prec, rmse, mae in eval_res]
    l_prec = [prec for nsrc, recall, prec, rmse, mae in eval_res]

    xlabel = 'Admissible Error (degree)'

    return (gen_figure('Recall', xlabel, names, etol, l_recall, xrange_=None),
            gen_figure('Precision', xlabel, names, etol, l_prec, xrange_=None))


def load_hmap_result(root,
                     name,
                     vad_suffix,
                     nb_size,
                     min_score=0.0,
                     anb_size=None,
                     gt_filter=None):
    """
    Load heat map and find local maxima as prediction together with ground truth.

    Note: different from load_rng_utt or load_rng_frame, the prediction
          comes with confidence score, which is the heat.

    Args:
        root       : path to database root
        name       : name of method
        vad_suffix : VAD ground truth file suffix
        nb_size    : size of neighborhood in radian
        min_score  : minimum score of possible prediction,
                     that is a threshold to find peaks
        anb_size   : azimuth neighbor size, if not None merge result on
                     azimuth direction
        gt_filter  : filter based on ground truth

    Returns:
        gt_pds : list of (id, ground_truth, prediction_with_score)
    """
    rdir = os.path.join(root, _RESULT_DIR, name)
    ddir = os.path.join(root, _GT_DIR)

    # load DOAs
    doa = np.load(os.path.join(rdir, 'doas.npy'))

    # find neigbors
    nlist = apkit.neighbor_list(doa, nb_size)

    gt_pds = []
    for infile in os.listdir(ddir):
        # iterate through all ground truth files
        if infile.endswith(_GT_SUFFIX):
            # seg id
            sid = infile[:-len(_GT_SUFFIX)]

            # list of ground truth per frame
            lgtf = utils.load_source_location_frame(root,
                                                    sid,
                                                    vad_suffix,
                                                    utils.LOAD_CERTAIN,
                                                    gt_filter=gt_filter)
        else:
            continue
        '''
        elif infile.endswith(_NEW_GT_SUFFIX):
            # seg id
            sid = infile[:-len(_NEW_GT_SUFFIX)]

            # list of ground truth per frame
            with open(os.path.join(ddir, infile), 'r') as f:
                lgtf = pickle.load(f)
        '''

        # id of frames with annotation (namely those are not edge cases)
        valid_fid = [fid for fid, _ in lgtf]

        # load heat values
        phi = np.load(os.path.join(rdir, sid + '.npy'))
        phi = phi[:, valid_fid]

        # find peaks
        pred = apkit.local_maxima(phi, nlist, th_phi=min_score)

        if anb_size is not None:
            pred = apkit.merge_lm_on_azimuth(phi, pred, doa, anb_size)

        for ldoa, p, (_, gt_locs) in zip(pred, phi.T, lgtf):
            gt_pds.append((None, gt_locs, [(doa[i], p[i]) for i in ldoa]))

    return gt_pds


def load_hmap_max(root, name, vad_suffix):
    """Load heat map result and find max as prediction
       (assume one source is present)

    Args:
        root     : path to database root
        name     : name of method
        vad_suffix : VAD ground truth file suffix.
        non_sil  : non-silent frames only

    Returns:
        res_gt : list of (id, ground_truth, prediction),
                 where id is "SEG_ID:FRAME_ID"
    """
    rdir = os.path.join(root, _RESULT_DIR, name)
    ddir = os.path.join(root, _GT_DIR)

    # load DOAs
    doa = np.load(os.path.join(rdir, 'doas.npy'))

    gt_pds = []
    for infile in os.listdir(ddir):
        # iterate through all ground truth files
        if infile.endswith(_GT_SUFFIX):
            # seg id
            sid = infile[:-len(_GT_SUFFIX)]

            # list of ground truth per frame
            lgtf = utils.load_source_location_frame(root, sid, vad_suffix,
                                                    utils.LOAD_ONE_SOURCE)

            # load heat values
            phi = np.load(os.path.join(rdir, sid + '.npy'))

            # find argmax
            max_ind = np.argmax(phi, axis=0)

            for fid, gt_locs in lgtf:
                gt_pds.append(
                    ('%s:%d' % (sid, fid), gt_locs, [doa[max_ind[fid]]]))

    return gt_pds


def find_thresholds(gt_pds, n):
    """Find n reference thresholds, that evenly separate all samples.

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result
        n      : number of desired thresholds

    Returns:
        ths    : list of threshold, the number of elements is decided by
                 min(n, unique scores)
    """
    scores = np.asarray([s for _, _, pred in gt_pds for _, s in pred])
    uniq_sorted = np.unique(scores)

    m = len(uniq_sorted)
    if m <= n:
        return uniq_sorted
    else:
        indices = np.arange(n, dtype=int) * m // n
        return uniq_sorted[indices]


def append_common_th(ths):
    return sorted(list(ths) + [x / 10.0 for x in range(1, 10)])


def pds_to_pdf(gt_pds, th):
    """Convert prediction with score to fixed,
       by selecting predictions with
                        score >= threshold

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result
        th     : threshold

    Returns:
        res_gt : list of (id, ground_truth, prediction)
    """
    return [(sid, gt_locs, [doa for doa, score in pred if score >= th])
            for sid, gt_locs, pred in gt_pds]


def pds_to_pdf_known_nsrc(gt_pds):
    """Convert prediction with score to fixed, by selecting predictions
       with N highest scores, where N is the number sources in the
       ground truth.

       If the number of predictions is 0, fill with (1,0,0).
       If the number of predictions is less than number of sources,
       duplicate the first prediction (largest score).
       Thus, the output number of predictions should always be equal to
       the number of sources.

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result

    Returns:
        res_gt : list of (id, ground_truth, prediction)
    """
    def _highest_k_pred(pred, k):
        if len(pred) == 0:
            return [(1.0, 0.0, 0.0)] * k
        # highest k predictions
        hk = sorted(pred, reverse=True, key=lambda x: x[1])[:k]
        while len(hk) < k:
            hk.append(hk[0])
        assert len(hk) == k
        return [doa for doa, score in hk]

    return [(sid, gt_locs, _highest_k_pred(pred, len(gt_locs)))
            for sid, gt_locs, pred in gt_pds]


def rp_various_th(gt_pds, ths, etol, metric=azimuth_distance):
    """Compute precision and recall of different score threshold

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result
        ths    : list of score thresholds
        etol   : admissible error

    Returns:
        rp     : list of (recall, precision)
    """
    pr = []
    for t in ths:
        pdf = pds_to_pdf(gt_pds, t)
        _, r, p, _, _ = eval_recall_precision(pdf, [etol], metric=metric)
        pr.append((r[0], p[0]))
    return pr


def _f1_score(rp):
    """Compute F1 score

    Args:
        rp : list of (recall, precision)

    Returns:
        f1 : list of F1 scores
    """
    return [2.0 * r * p / (r + p) if r * p > 0.0 else 0.0 for (r, p) in rp]


def gen_p_r_figures(l_gt_pds,
                    names,
                    etol,
                    title=None,
                    best_f1=False,
                    metric=azimuth_distance):
    """Generate precision and recall figures' gnuplot script

    Args:
        l_gt_pds : list of gt and prediction with scores
        names    : list of method names
        etol     : admissible error

    Returns:
        plot     : script (as str) of plot
        f1_info  : (if best_f1) list of best f1 score and corresponding
                   threshold, recall and precision: (th, f1, recall, prec)
    """
    ths = [append_common_th(find_thresholds(rg, 200)[-50:]) for rg in l_gt_pds]
    rps = [
        np.asarray(rp_various_th(rg, th, etol, metric=metric))
        for rg, th in zip(l_gt_pds, ths)
    ]
    l_recall = [rp[:, 0] for rp in rps]
    l_prec = [rp[:, 1] for rp in rps]

    if title is None:
        title = 'Precision vs Recall (Adm. Error=%d degrees)' \
                    % int(etol / math.pi * 180)

    fig = gen_figure('Precision',
                     'Recall',
                     names,
                     l_recall,
                     l_prec,
                     data_opt='w l',
                     title=title)
    if best_f1:
        f1s = [_f1_score(rp) for rp in rps]
        best_ids = [np.argmax(f1) for f1 in f1s]
        f1_info = [(th[i], f1[i], rp[i][0], rp[i][1])
                   for i, th, f1, rp in zip(best_ids, ths, f1s, rps)]
        return fig, f1_info
    else:
        return fig


def gen_report_known_nsrc(gt_pds, etol, title=''):
    """Generate performance report of predition with known number of
       sources

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result
        etol   : admissible error (for ACC)
    """
    report = [title]

    gt_pdf = pds_to_pdf_known_nsrc(gt_pds)

    # split by number of sources
    nmax = max(len(gt_locs) for _, gt_locs, pred in gt_pdf)
    gt_pdf_by_nsrc = [[] for _ in range(nmax)]

    for entry in gt_pdf:
        nsrc = len(entry[1])
        if nsrc > 0:
            gt_pdf_by_nsrc[nsrc - 1].append(entry)

    # eval all subsets
    mae_by_nsrc = []
    acc_by_nsrc = []
    for subset in gt_pdf_by_nsrc:
        _, recall, _, _, mae = eval_recall_precision(subset, [etol, math.pi])
        mae_by_nsrc.append(mae[1])
        acc_by_nsrc.append(recall[0])

    # overal result
    nframe_by_nsrc = [len(subset) for subset in gt_pdf_by_nsrc]
    mae_overall = np.average(mae_by_nsrc, weights=nframe_by_nsrc)
    acc_overall = np.average(acc_by_nsrc, weights=nframe_by_nsrc)

    report.append('Overall (%d frames)' % sum(nframe_by_nsrc))
    report.append('  MAE       : %.1f degree' \
                        % (mae_overall / _ONE_DEGREE))
    report.append('  ACC (e=%d) : %.1f%%' \
                        % (int(etol / _ONE_DEGREE), acc_overall * 100.0))

    latex_mae = [
        '$%.1f$' % (m / _ONE_DEGREE) for m in [mae_overall] + mae_by_nsrc
    ]
    latex_acc = ['$%.1f$' % (a * 100.0) for a in [acc_overall] + acc_by_nsrc]
    latex = ' & '.join([x for y in zip(latex_mae, latex_acc) for x in y])
    report.append(latex)

    for i in range(nmax):
        report.append('NSRC=%d (%d frames)' % (i + 1, nframe_by_nsrc[i]))
        report.append('  MAE       : %.1f degree' \
                            % (mae_by_nsrc[i] / _ONE_DEGREE))
        report.append('  ACC (e=%d) : %.1f%%' \
                            % (int(etol / _ONE_DEGREE), acc_by_nsrc[i] * 100.0))

    # split by elevation, 5 sections between -90, -60, -30, 0, 30, 60, 90
    eranges = [
        'ELE < -60', 'ELE in [-60,-30)', 'ELE in [-30,  0)',
        'ELE in [  0, 30)', 'ELE in [ 30, 60)', 'ELE > 60'
    ]
    gt_pdf_by_ele = [[] for i in eranges]

    for entry in gt_pdf_by_nsrc[0]:
        z = entry[1][0][2]
        if z < math.sin(-60 * _ONE_DEGREE):
            gt_pdf_by_ele[0].append(entry)
        elif z < math.sin(-30 * _ONE_DEGREE):
            gt_pdf_by_ele[1].append(entry)
        elif z < 0:
            gt_pdf_by_ele[2].append(entry)
        elif z < math.sin(30 * _ONE_DEGREE):
            gt_pdf_by_ele[3].append(entry)
        elif z < math.sin(60 * _ONE_DEGREE):
            gt_pdf_by_ele[4].append(entry)
        else:
            gt_pdf_by_ele[5].append(entry)

    for subset, erange in zip(gt_pdf_by_ele, eranges):
        if len(subset) > 0:
            _, recall, _, _, mae = eval_recall_precision(
                subset, [etol, math.pi])
            report.append('%s (%d frames)' % (erange, len(subset)))
            report.append('  MAE       : %.1f degree' \
                                % (mae[1] / _ONE_DEGREE))
            report.append('  ACC (e=%d) : %.1f%%' \
                                % (int(etol / _ONE_DEGREE), recall[0] * 100.0))

    # split by azimuth abs, 6 sections: 0, 30, 60, 90, 120, 150 180
    aranges = [
        '|AZI| <  30', '|AZI| in [ 30,  60)', '|AZI| in [ 60,  90)',
        '|AZI| in [ 90, 120)', '|AZI| in [120, 150)', '|AZI| >= 150'
    ]
    gt_pdf_by_azi = [[] for i in aranges]

    for entry in gt_pdf_by_nsrc[0]:
        x, y, _ = entry[1][0]
        xnorm = x / math.sqrt(x * x + y * y)
        if xnorm > math.cos(30 * _ONE_DEGREE):
            gt_pdf_by_azi[0].append(entry)
        elif xnorm > math.cos(60 * _ONE_DEGREE):
            gt_pdf_by_azi[1].append(entry)
        elif xnorm > 0:
            gt_pdf_by_azi[2].append(entry)
        elif xnorm > math.cos(120 * _ONE_DEGREE):
            gt_pdf_by_azi[3].append(entry)
        elif xnorm > math.cos(150 * _ONE_DEGREE):
            gt_pdf_by_azi[4].append(entry)
        else:
            gt_pdf_by_azi[5].append(entry)

    for subset, arange in zip(gt_pdf_by_azi, aranges):
        if len(subset) > 0:
            _, recall, _, _, mae = eval_recall_precision(
                subset, [etol, math.pi])
            report.append('%s (%d frames)' % (arange, len(subset)))
            report.append('  MAE       : %.1f degree' \
                                % (mae[1] / _ONE_DEGREE))
            report.append('  ACC (e=%d) : %.1f%%' \
                                % (int(etol / _ONE_DEGREE), recall[0] * 100.0))
    return '\n'.join(report)


def gen_error_per_frame(gt_pds, metric=azimuth_distance):
    """List absolute error per sound source assuming known number of sources

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result
    """
    report = []

    gt_pdf = pds_to_pdf_known_nsrc(gt_pds)

    for sid, gt, pred in gt_pdf:
        match = match_gt(pred, gt, metric, math.pi)
        nsrc = len(gt)
        for _, (_, dis) in match.items():
            report.append('%d\t%.3f' % (nsrc, dis * 180.0 / math.pi))

    return '\n'.join(report)


def gen_report_unknown_nsrc(gt_pds, th, etol, title=''):
    """Generate performance report of predition with unknown number of
       sources

    Args:
        gt_pds : list of (id, ground_truth, prediction_with_score),
                 loaded by load_hmap_result
        th     : threshold (should be the optimal one)
        etol   : admissible error
    """
    report = [
        title,
        '  Admissible Error = %d degree' % int(etol / _ONE_DEGREE)
    ]

    gt_pdf = pds_to_pdf(gt_pds, th)

    # split by number of sources
    nmax = max(len(gt_locs) for _, gt_locs, pred in gt_pdf)
    gt_pdf_by_nsrc = [[] for _ in range(nmax)]

    for entry in gt_pdf:
        nsrc = len(entry[1])
        if nsrc > 0:
            gt_pdf_by_nsrc[nsrc - 1].append(entry)

    # eval all subsets
    latex = []
    for i, subset in enumerate(gt_pdf_by_nsrc):
        _, recall, prec, _, _ = eval_recall_precision(subset, [etol])
        report.append('NSRC=%d (%d frames)' % (i + 1, len(subset)))
        report.append('  RECALL    : %.2f' % recall[0])
        report.append('  PRECISION : %.2f' % prec[0])
        report.append('  F1-SCORE  : %.2f' %
                      _f1_score([(recall[0], prec[0])])[0])
        latex.append(
            '$%.2f$ & $%.2f$ & $%.2f$' %
            (recall[0], prec[0], _f1_score([(recall[0], prec[0])])[0]))
    report.append(' & '.join(latex))

    # split by source separation
    if nmax >= 2:
        _RANGES = [(i, i + 10) for i in range(0, 180, 10)]
        _RANGES_STR = ['%d~%d' % r for r in _RANGES]
        _GT_FILTERS = [
            utils.SrcDistRangeFilter(a * _ONE_DEGREE, b * _ONE_DEGREE)
            for a, b in _RANGES
        ]
        for gt_filter, srange in zip(_GT_FILTERS, _RANGES_STR):
            subset = [
                entry for entry in gt_pdf_by_nsrc[1] if gt_filter(entry[1])
            ]
            if len(subset) == 0:
                continue
            _, recall, prec, _, _ = eval_recall_precision(subset, [etol])
            report.append('%s (%d frames)' % (srange, len(subset)))
            report.append('  RECALL    : %.2f' % recall[0])
            report.append('  PRECISION : %.2f' % prec[0])
            report.append('  F1-SCORE  : %.2f' %
                          _f1_score([(recall[0], prec[0])])[0])
    return '\n'.join(report)


def load_2tasks_heat(rdir, sid, fid_filter=None):
    """
    Load head map of a single file without ground truth.

    Args:
        rdir       : path to the results directory
        sid        : sequence ID
        fid_filter : (default None) frame index filter if not None

    Returns:
        phi : SSL heat map, indexed by (dt)
        sns : SNS heat map, indexed by (dt)
    """
    pred = np.load(os.path.join(rdir, sid + '.npy'))
    if pred.ndim == 2 or pred.shape[2] == 1:
        if pred.ndim == 3:
            phi = pred[:, :, 0]
        else:
            phi = pred
        snsfile = os.path.join(rdir, sid + '.sns.npy')
        if os.path.isfile(snsfile):
            sns = np.load(snsfile)
        else:
            sns = np.ones(phi.shape)
    else:
        phi = pred[:, :, 0]
        sns = pred[:, :, 1]

    if fid_filter is not None:
        phi = phi[:, fid_filter]
        sns = sns[:, fid_filter]

    return phi, sns


def load_2tasks_result(root,
                       name,
                       win_size,
                       hop_size,
                       nb_size,
                       min_score=0.0,
                       anb_size=None,
                       gt_filter=None):
    """
    Load heat map and find local maxima as prediction together with ground truth.

    Note: different from load_rng_utt or load_rng_frame, the prediction
          comes with confidence score, which is the heat.

    Args:
        root       : path to database root
        name       : name of method
        win_size   : window size
        hop_size   : hop size
        nb_size    : size of neighborhood in radian
        min_score  : minimum score of possible prediction,
                     that is a threshold to find peaks
        anb_size   : azimuth neighbor size, if not None merge result on
                     azimuth direction
        gt_filter  : filter based on ground truth

    Returns:
        gt_pds : list of (id, ground_truth, prediction_with_score)
    """
    rdir = os.path.join(root, _RESULT_DIR, name)

    # load DOAs
    doa = np.load(os.path.join(rdir, 'doas.npy'))

    # find neigbors
    nlist = apkit.neighbor_list(doa, nb_size)

    gt_pds = []
    for sid in utils.all_sids(root):
        # list of ground truth per frame
        lgtf = utils.load_gtf(root, sid, win_size, hop_size)

        # id of frames with annotation (namely those are not edge cases)
        valid_fid = [fid for fid, _ in lgtf]

        # load heat values
        phi, sns = load_2tasks_heat(rdir, sid, fid_filter=valid_fid)

        # find peaks
        pred = apkit.local_maxima(phi, nlist, th_phi=min_score)

        if anb_size is not None:
            pred = apkit.merge_lm_on_azimuth(phi, pred, doa, anb_size)

        for ldoa, p, s, (_, gt) in zip(pred, phi.T, sns.T, lgtf):
            for i in ldoa:
                assert not np.isnan(s[i])
            gt_pds.append((None, gt, [(doa[i], p[i], s[i]) for i in ldoa]))

    return gt_pds


def _doa_index(doas, loc):
    dis = scipy.spatial.distance.cdist(doas, [loc], metric=angular_distance)
    return np.argmin(dis)


def load_speech_nonspeech_result(root,
                                 name,
                                 win_size,
                                 hop_size,
                                 gt_filter=None):
    """
    Load speech/non-speech result assuming sources are correcly localized

    Args:
        root       : path to database root
        name       : name of method
        win_size   : window size
        hop_size   : hop size
        gt_filter  : filter based on ground truth

    Returns:
        gt_pds : list of (ground_truth, predition_score)
    """
    rdir = os.path.join(root, _RESULT_DIR, name)
    ddir = os.path.join(root, _GT_DIR)

    # load DOAs
    doa = np.load(os.path.join(rdir, 'doas.npy'))

    gt_pds = []
    for infile in os.listdir(ddir):
        # iterate through all ground truth files
        if infile.endswith(_WAV_SUFFIX):
            # seg id
            sid = infile[:-len(_WAV_SUFFIX)]

            # list of ground truth per frame
            lgtf = utils.load_gtf(root, sid, win_size, hop_size)
        else:
            continue

        pred = np.load(os.path.join(rdir, sid + '.npy'))
        for fid, gt in lgtf:
            sns = pred[:, fid, 1]

            for loc, stype, _ in gt:
                di = _doa_index(doa, loc)
                gt_pds.append((stype, sns[di]))

    return gt_pds


def load_sns_gtdoa_results(root, name, win_size, hop_size, sns_task_id):
    """
    Load SNS predictions on ground truth DOAs from multi-task nn results.

    Args:
        root : path to dataset
        name : method name
        win_size : you know what it is
        hop_size : you know what it is
        sns_task_id : speech/non-speech task id

    Returns:
        sns_res_wgt : list of (ground_truth, predition_score, orig_gt)
    """
    # load DOAs
    doas = utils.load_doa(root, name)

    # sns result with groudn truth
    sns_res_wgt = []

    # iterate all files
    for sid in utils.all_sids(root):
        # ground truth per frame
        lgtf = utils.load_gtf(root,
                              sid,
                              win_size,
                              hop_size,
                              gtf_dir='gt_frame')

        # SNS prediction
        sns = utils.load_pred(root, name, sid, task_id=sns_task_id)
        if sns.ndim == 3:
            assert sns.shape[1] == 1
            sns = np.squeeze(sns, axis=1)
        assert sns.ndim == 2

        for fid, gt in lgtf:
            for gt_loc, gt_stype, _ in gt:
                loc_id = loc_to_id(gt_loc, doas)
                sns_res_wgt.append((gt_stype, sns[fid, loc_id], gt))
    return sns_res_wgt


def load_sns_pddoa_results(gt_pds,
                           etol,
                           metric=azimuth_distance,
                           min_score=0.5):
    """
    Load SNS results on predited DOAs

    Args:
        gt_pds  : 2tasks results loaded by load_2tasks_result
        etol    : admissible error

    Returns:
        sns_res_wgt : list of (ground_truth, predition_score, orig_gt)
    """
    sns_res_wgt = []
    for _, gt, pd in gt_pds:
        gtloc = [loc for loc, _, _ in gt]
        pdloc = [loc for loc, score, _ in pd if score >= min_score]
        match = match_gt(pdloc, gtloc, metric, etol)
        for gid, (pid, _) in list(match.items()):
            sns_res_wgt.append((gt[gid][1], pd[pid][2], gt))
    return sns_res_wgt


def filter_sns_result_by_gt(sns_res_wgt, filt=(lambda x: True)):
    return [(gt, pd) for gt, pd, orig_gt in sns_res_wgt if filt(orig_gt)]


def eval_sns_perf(sns_res, threshold=0.5):
    """
    Returns:
        true positive,
        true negative,
        false positive,
        false negative,
        accuracy,
        true positive rate (recall),
        false positive rate
    """
    gt = np.array([g for g, p in sns_res]) == 1.0
    pd = np.array([p for g, p in sns_res]) > 0.5

    assert gt.shape == pd.shape

    tp = np.sum(gt * pd)
    tn = np.sum((1.0 - gt) * (1.0 - pd))
    fp = np.sum((1.0 - gt) * pd)
    fn = np.sum(gt * (1.0 - pd))

    total = len(gt)
    assert (tp + tn + fp + fn) == total

    return (tp, tn, fp, fn, 1.0 * (tp + tn) / total, 1.0 * tp / (tp + fn),
            1.0 * fp / (fp + tn))


def sns_perf_to_text(sns_perf):
    #tp, tn, fp, fn, acc, tpr, fpr = sns_perf
    return 'tp=%d, tn=%d, fp=%d, fn=%d, acc=%.3f, tpr=%.3f, fpr=%.3f' % sns_perf


def gen_report_sns(sns_res_wgt, title='', orig_rg=None):
    report = [title]

    report.append('')
    sns_res = filter_sns_result_by_gt(sns_res_wgt)
    if orig_rg is None:
        report.append('== Overall (%d frames) ==' % len(sns_res))
    else:
        total_gtsrc = sum([len(gt) for _, gt, pd in orig_rg])
        report.append('== Overall (%d frames, %.2f recall) ==' %
                      (len(sns_res), 1.0 * len(sns_res) / total_gtsrc))
    report.append(sns_perf_to_text(eval_sns_perf(sns_res)))

    report.append('')
    filt = utils.SrcNoiseMixFilter
    sns_res = filter_sns_result_by_gt(sns_res_wgt, filt)
    if orig_rg is None:
        report.append('== Mixed (%d frames) ==' % len(sns_res))
    else:
        total_gtsrc = sum([len(gt) for _, gt, pd in orig_rg if filt(gt)])
        report.append('== Mixed (%d frames, %.2f recall) ==' %
                      (len(sns_res), 1.0 * len(sns_res) / total_gtsrc))
    report.append(sns_perf_to_text(eval_sns_perf(sns_res)))

    report.append('')
    filt = utils.ReverseFilter(utils.SrcNoiseMixFilter)
    sns_res = filter_sns_result_by_gt(sns_res_wgt, filt)
    if orig_rg is None:
        report.append('== Only source or only noise (%d frames) ==' %
                      len(sns_res))
    else:
        total_gtsrc = sum([len(gt) for _, gt, pd in orig_rg if filt(gt)])
        report.append(
            '== Only source or only noise (%d frames, %.2f recall) ==' %
            (len(sns_res), 1.0 * len(sns_res) / total_gtsrc))
    report.append(sns_perf_to_text(eval_sns_perf(sns_res)))

    return '\n'.join(report)


def load_bf_sns_gtdoa_results(root, name, bf_name):
    """
    Load speech/non-speech result assuming sources are correcly localized

    Args:
        root    : path to database root
        name    : name of method
        bf_name : name of beamforming method

    Returns:
        gt_pds : list of (ground_truth, predition_score)
    """
    rdir = os.path.join(root, _BF_RESULT_DIR, name)
    ddir = os.path.join(root, _BF_DATA_DIR, bf_name)

    gt_pds = []
    for infile in os.listdir(ddir):
        # iterate through all bf ground truth
        if infile.endswith(_STYPE_SUFFIX):
            # seg id
            sid = infile[:-len(_STYPE_SUFFIX)]

            # list of ground truth per frame
            with open(os.path.join(ddir, infile), 'r') as f:
                sns_gt = pickle.load(f)

            pred = np.load(os.path.join(rdir, sid + '.npy'))
            gt_pds += list(zip(sns_gt, pred[:, 1]))

    return gt_pds


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
