#!/usr/bin/env python
"""
gen_2tasks_report.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys
import os
import argparse
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common import utils, evaluation

# DEBUG
import time


def _join(list_of_lists):
    return [x for l in list_of_lists for x in l]


def _as_ssl(rg, filt=None):
    return [(
        None,
        [loc for loc, _, _ in gt],
        [(doa, phi) for doa, phi, _ in pred],
    ) for _, gt, pred in rg if (True if filt is None else filt(gt))]


def _as_speech(rg):
    return [(
        None,
        [loc for loc, stype, _ in gt if stype == 1],
        [(doa, phi * sns) for doa, phi, sns in pred],
    ) for _, gt, pred in rg]


def _as_speech_filter(rg, filt):
    return [(
        None,
        [loc for loc, stype, _ in gt if stype == 1],
        [(doa, phi * sns) for doa, phi, sns in pred],
    ) for _, gt, pred in rg if filt(gt)]


def main(test_path, method, win_size, hop_size, etol, nb_size, min_sc,
         snsperf_min_sc, anb_size, output, ssl_only, no_pr_plot, detail_error):
    s = time.time()

    rg = _join([
        evaluation.load_2tasks_result(
            p,
            method,
            win_size,
            hop_size,
            nb_size,
            min_sc,
            anb_size,
        ) for p in test_path
    ])

    t = time.time()
    print('data loaded: %.3f' % (t - s), file=sys.stderr)
    print('data samples: %d' % len(rg), file=sys.stderr)
    s = t

    if output != '.':
        os.makedirs(output)

    if not ssl_only:
        # eval SNS classification on predicted DOAs
        sns_res_wgt = evaluation.load_sns_pddoa_results(
            rg, etol, min_score=snsperf_min_sc)

        t = time.time()
        print('as speech/non-speech classification: %.3f' % (t - s),
              file=sys.stderr)
        s = t

        with open(os.path.join(output, 'sns_pddoa'), 'w') as f:
            print(evaluation.gen_report_sns(sns_res_wgt, method, rg), file=f)
            print('', file=f)
            print('# localization threshold : %.2f' % snsperf_min_sc, file=f)

        t = time.time()
        print('gen sns pddoa: %.3f' % (t - s), file=sys.stderr)
        s = t

    # as SSL
    rg_ssl = _as_ssl(rg)

    t = time.time()
    print('as sound source localization: %.3f' % (t - s), file=sys.stderr)
    s = t

    with open(os.path.join(output, 'ssl_known_nsrc'), 'w') as f:
        print(evaluation.gen_report_known_nsrc(rg_ssl, etol, method), file=f)

    t = time.time()
    print('gen known nsrc: %.3f' % (t - s), file=sys.stderr)
    s = t

    if detail_error:
        with open(os.path.join(output, 'ssl_error_per_frame'), 'w') as f:
            print(evaluation.gen_error_per_frame(rg_ssl), file=f)

        t = time.time()
        print('print error per frame: %.3f' % (t - s), file=sys.stderr)
        s = t

    if no_pr_plot:
        # stop here
        return

    plot, f1_info = evaluation.gen_p_r_figures([rg_ssl], [method],
                                               etol,
                                               best_f1=True)
    th, f1, recall, prec = f1_info[0]

    with open(os.path.join(output, 'ssl_pr_plot'), 'w') as f:
        print(plot, file=f)

    t = time.time()
    print('gen pr plot: %.3f' % (t - s), file=sys.stderr)
    s = t

    with open(os.path.join(output, 'ssl_unknown_nsrc'), 'w') as f:
        print(f1_info, file=f)
        print(evaluation.gen_report_unknown_nsrc(rg_ssl, th, etol, method),
              file=f)

    t = time.time()
    print('gen unknown nsrc: %.3f' % (t - s), file=sys.stderr)
    s = t

    for nsrc in range(1, 5):
        filt = utils.NSrcFilter(nsrc)
        rg_sub = _as_ssl(rg, filt)

        if len(rg_sub) == 0:
            continue

        print('# frames s%d: %d' % (nsrc, len(rg_sub)), file=sys.stderr)
        plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

        with open(os.path.join(output, 'ssl_pr_plot_s%d' % nsrc), 'w') as f:
            print(plot, file=f)

        t = time.time()
        print('gen pr plot s%d: %.3f' % (nsrc, t - s), file=sys.stderr)
        s = t

    for nnsrc in range(0, 3):
        filt = utils.NNoiseFilter(nnsrc)
        rg_sub = _as_ssl(rg, filt)

        if len(rg_sub) == 0:
            continue

        print('# frames n%d: %d' % (nnsrc, len(rg_sub)), file=sys.stderr)
        plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

        with open(os.path.join(output, 'ssl_pr_plot_n%d' % nnsrc), 'w') as f:
            print(plot, file=f)

        t = time.time()
        print('gen pr plot n%d: %.3f' % (nnsrc, t - s), file=sys.stderr)
        s = t

    if not ssl_only:
        # as speech detection
        rg_speech = _as_speech(rg)

        t = time.time()
        print('as speech source localization: %.3f' % (t - s), file=sys.stderr)
        s = t

        with open(os.path.join(output, 'speech_known_nsrc'), 'w') as f:
            print(evaluation.gen_report_known_nsrc(rg_speech, etol, method),
                  file=f)

        t = time.time()
        print('gen known nsrc: %.3f' % (t - s), file=sys.stderr)
        s = t

        plot, f1_info = evaluation.gen_p_r_figures([rg_speech], [method],
                                                   etol,
                                                   best_f1=True)
        th, f1, recall, prec = f1_info[0]

        with open(os.path.join(output, 'speech_pr_plot'), 'w') as f:
            print(plot, file=f)

        t = time.time()
        print('gen pr plot: %.3f' % (t - s), file=sys.stderr)
        s = t

        with open(os.path.join(output, 'speech_unknown_nsrc'), 'w') as f:
            print(f1_info, file=f)
            print(evaluation.gen_report_unknown_nsrc(rg_speech, th, etol,
                                                     method),
                  file=f)

        t = time.time()
        print('gen unknown nsrc: %.3f' % (t - s), file=sys.stderr)
        s = t

        filt = utils.NNoiseFilter(0)
        rg_sub = _as_speech_filter(rg, filt)
        print('# frames n0: %d' % len(rg_sub), file=sys.stderr)

        if len(rg_sub) > 0 and len(rg_sub) < len(rg):
            plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

            with open(os.path.join(output, 'speech_pr_plot_n0'), 'w') as f:
                print(plot, file=f)

        t = time.time()
        print('gen pr plot n0: %.3f' % (t - s), file=sys.stderr)
        s = t

        filt = utils.NNoiseFilter(1)
        rg_sub = _as_speech_filter(rg, filt)
        print('# frames n1: %d' % len(rg_sub), file=sys.stderr)

        if len(rg_sub) > 0 and len(rg_sub) < len(rg):
            plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

            with open(os.path.join(output, 'speech_pr_plot_n1'), 'w') as f:
                print(plot, file=f)

        t = time.time()
        print('gen pr plot n1: %.3f' % (t - s), file=sys.stderr)
        s = t
    '''
    filt = utils.NSrcNoiseFilter(2, 0)
    rg_sub = _as_speech_filter(rg, filt)

    if len(rg_sub) > 0:
        plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

        with open(os.path.join(output, 'speech_pr_plot_s2_n0'), 'w') as f:
            print >> f, plot

    t = time.time()
    print >> sys.stderr, 'gen pr plot s2 n0: %.3f' % (t - s)
    s = t

    filt = utils.NSrcNoiseFilter(1, 1)
    rg_sub = _as_speech_filter(rg, filt)

    if len(rg_sub) > 0:
        plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

        with open(os.path.join(output, 'speech_pr_plot_s1_n1'), 'w') as f:
            print >> f, plot

    t = time.time()
    print >> sys.stderr, 'gen pr plot s1 n1: %.3f' % (t - s)
    s = t

    filt = utils.NSrcNoiseFilter(2, 1)
    rg_sub = _as_speech_filter(rg, filt)

    if len(rg_sub) > 0:
        plot = evaluation.gen_p_r_figures([rg_sub], [method], etol)

        with open(os.path.join(output, 'speech_pr_plot_s2_n1'), 'w') as f:
            print >> f, plot

    t = time.time()
    print >> sys.stderr, 'gen pr plot s2 n1: %.3f' % (t - s)
    s = t
    '''

    # as non-speech detection
    '''
    rg_nonspeech = _as_nonspeech(rg)

    t = time.time()
    print >> sys.stderr, 'as sound source localization: %.3f' % (t - s)
    s = t

    with open(os.path.join(output, 'nonspeech_known_nsrc'), 'w') as f:
        print >> f, evaluation.gen_report_known_nsrc(rg_nonspeech, etol, method)

    t = time.time()
    print >> sys.stderr, 'gen known nsrc: %.3f' % (t - s)
    s = t

    plot, f1_info = evaluation.gen_p_r_figures([rg_nonspeech], [method], etol,
                                               best_f1=True)
    th, f1, recall, prec = f1_info[0]

    with open(os.path.join(output, 'nonspeech_pr_plot'), 'w') as f:
        print >> f, plot

    t = time.time()
    print >> sys.stderr, 'gen pr plot: %.3f' % (t - s)
    s = t

    with open(os.path.join(output, 'nonspeech_unknown_nsrc'), 'w') as f:
        print >> f, f1_info
        print >> f, evaluation.gen_report_unknown_nsrc(rg_nonspeech, th, etol, method)

    t = time.time()
    print >> sys.stderr, 'gen unknown nsrc: %.3f' % (t - s)
    s = t

    filt = utils.NSrcFilter(1)
    rg_n1 = [x for x in rg_nonspeech if filt(x[1])]

    plot = evaluation.gen_p_r_figures([rg_n1], [method], etol)

    with open(os.path.join(output, 'nonspeech_pr_plot_n1'), 'w') as f:
        print >> f, plot

    t = time.time()
    print >> sys.stderr, 'gen pr plot n1: %.3f' % (t - s)
    s = t

    filt = utils.NSrcFilter(2)
    rg_n1 = [x for x in rg_nonspeech if filt(x[1])]

    plot = evaluation.gen_p_r_figures([rg_n1], [method], etol)

    with open(os.path.join(output, 'nonspeech_pr_plot_n2'), 'w') as f:
        print >> f, plot

    t = time.time()
    print >> sys.stderr, 'gen pr plot n2: %.3f' % (t - s)
    s = t
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate performance report')
    parser.add_argument('test',
                        metavar='TEST_PATH',
                        type=str,
                        nargs='+',
                        help='path to test data and feature')
    parser.add_argument('-m',
                        '--method',
                        metavar='METHOD',
                        type=str,
                        required=True,
                        help='method name')
    parser.add_argument('-w',
                        '--window-size',
                        metavar='WIN_SIZE',
                        type=int,
                        default=2048,
                        help='(default 2048) analysis window size')
    parser.add_argument('-o',
                        '--hop-size',
                        metavar='HOP_SIZE',
                        type=int,
                        default=1024,
                        help='(default 1024) hop size, '
                        'number of samples between windows')
    parser.add_argument('-e',
                        '--adm-error',
                        metavar='ERROR',
                        type=int,
                        default=5,
                        help='(default 5) admissible error in degrees')
    parser.add_argument('--neighbor-size',
                        metavar='SIZE',
                        type=int,
                        default=8,
                        help='(default 8) neighborhood size in degrees')
    parser.add_argument('--min-score',
                        metavar='SCORE',
                        type=float,
                        default=0.0,
                        help='(default 0.0) minimun score for peaks')
    parser.add_argument(
        '--snsperf-min-score',
        metavar='SCORE',
        type=float,
        default=0.5,
        help='(default 0.5) minimun score for '
        'peak finding in evaluation of sns performance on predicted DOAs')
    parser.add_argument(
        '--azi-neighbor-size',
        metavar='SIZE',
        type=int,
        default=None,
        help='(default None) azimuth neighborhood size in degrees')
    parser.add_argument('--output',
                        metavar='PATH',
                        type=str,
                        default='.',
                        help='output directory, new directory will be '
                        'created if not exist')
    parser.add_argument('--ssl-only',
                        action='store_true',
                        help='evaluate SSL performance only')
    parser.add_argument('--no-pr-plot',
                        action='store_true',
                        help='do not generate precision-recall plot')
    parser.add_argument('--detail-error',
                        action='store_true',
                        help='print error (distance between prediction and '
                        'groundtruth assuming known number of sources) '
                        'per source')
    args = parser.parse_args()
    main(
        args.test, args.method, args.window_size, args.hop_size,
        args.adm_error * math.pi / 180, args.neighbor_size * math.pi / 180,
        args.min_score, args.snsperf_min_score, args.azi_neighbor_size *
        math.pi / 180 if args.azi_neighbor_size is not None else None,
        args.output, args.ssl_only, args.no_pr_plot, args.detail_error)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
