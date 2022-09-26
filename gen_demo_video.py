#!/usr/bin/env python
"""
gen_demo_video.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>

This file is part of "Neural Network based Sound Source Localization Models".

"Neural Network based Sound Source Localization Models" is free software:
you can redistribute it and/or modify it under the terms of the BSD 3-Clause
License.

"Neural Network based Sound Source Localization Models" is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the BSD 3-Clause License for more details.
"""

import os
import argparse
import math
from fractions import Fraction
import pickle as pickle

import numpy as np
import cv2

import apkit

import evaluation

_CAM_MATRIX = np.array([592.779158393, 0.0, 328.225295327,
                        0.0, 591.991356581, 234.241795451,
                        0.0, 0.0, 1.0]).reshape((3,3))
_CAM_DIST = np.array([0.121716106149, -0.481610730654, -0.00144629247554,
                       0.00131082543473, 0.477385359915])

_CAM_R = np.array([0.0, -1.0,  0.0,
                   0.0,  0.0, -1.0,
                   1.0,  0.0,  0.0]).reshape((3,3))
_CAM_T = np.array([0.0, 0.163 - 0.2066, -0.087])

_HMAP_HEIGHT = 150
_X_EXPAND = 240

_HMAP_FOV_MARGIN = 20

_COLOR_PRED_SPEECH = (0,   100, 0  )
_COLOR_PRED_NOISE  = (60,  20,  220)
_COLOR_GT_SPEECH   = (204, 209, 72 )
_COLOR_GT_NOISE    = (200, 135, 255)
_COLOR_OUTPUT_SSL  = (255, 0,   0  )
_COLOR_OUTPUT_SNS  = (0,   140, 255)

_SNS_THRESHOLD = 0.5

_GT_PATTERN = '%s.w%d_o%d.gtf.pkl'

def qvfid(t, stamps):
    if t <= stamps[0]:
        return 0;
    elif t >= stamps[-1]:
        return len(stamps) - 1

    # binary search
    sid = 0
    eid = len(stamps)
    while eid - sid > 1:
        mid = (sid + eid) / 2
        if t >= stamps[mid]:
            sid = mid
        else:
            eid = mid
    assert eid == sid + 1
    return sid

def plot_hmap_fov(img, a2dx, phi, sns=None):
    height, width, _ = img.shape

    if sns is not None:
        pts = [(x, int(_HMAP_FOV_MARGIN 
                            + (height - 2 * _HMAP_FOV_MARGIN) 
                            * (1.0 - p))) for x, p in zip(a2dx, sns)
                                          if x is not None 
                                             and x >= -100 
                                             and x <= width + 100] 
        pts = sorted(pts, key=lambda x: x[0])
        cv2.polylines(img, np.asarray([pts]), False, _COLOR_OUTPUT_SNS, thickness=1,
                      lineType=cv2.CV_AA)

    pts = [(x, int(_HMAP_FOV_MARGIN 
                        + (height - 2 * _HMAP_FOV_MARGIN) 
                        * (1.0 - p))) for x, p in zip(a2dx, phi)
                                      if x is not None 
                                         and x >= -100 
                                         and x <= width + 100] 
    pts = sorted(pts, key=lambda x: x[0])
    cv2.polylines(img, np.asarray([pts]), False, _COLOR_OUTPUT_SSL, thickness=2,
                  lineType=cv2.CV_AA)

def plot_grid(img, a2dx):
    h, w, _ = img.shape
    ol = img.copy()
    for a in range(-25, 26, 5):
        lx = a2dx[a]
        assert lx is not None
        cv2.line(ol, (lx, 0), (lx, h), (160, 160, 160))
        cv2.putText(ol, '%d' % a, (lx, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1,
                    lineType=cv2.CV_AA)
    return 0.6 * img + 0.4 * ol

def plot_legend(img, t, sid, method_name, add_sns=False):
    h, w, _ = img.shape
    cv2.putText(img, 'Azimuth (degree)', (_X_EXPAND - 90, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                lineType=cv2.CV_AA)
    cv2.putText(img, 'Method: %s' % method_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                lineType=cv2.CV_AA)
    cv2.putText(img, 'File: %s' % sid, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                lineType=cv2.CV_AA)
    
    if not add_sns:
        cv2.putText(img, 'Ground truth:', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    lineType=cv2.CV_AA)
        cv2.circle(img, (170, 90), 10, _COLOR_GT_SPEECH, 2, lineType=cv2.CV_AA)

        cv2.putText(img, 'Prediction:', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    lineType=cv2.CV_AA)
        cv2.rectangle(img, (160, 115), (180, 135), _COLOR_PRED_SPEECH, -1)
    else:
        cv2.putText(img, 'G.T.  Speech:', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    lineType=cv2.CV_AA)
        cv2.circle(img, (170, 93), 10, _COLOR_GT_SPEECH, 2, lineType=cv2.CV_AA)

        cv2.putText(img, 'G.T.  Noise:', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    lineType=cv2.CV_AA)
        cv2.circle(img, (170, 123), 10, _COLOR_GT_NOISE, 2, lineType=cv2.CV_AA)

        cv2.putText(img, 'Pred. Speech:', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    lineType=cv2.CV_AA)
        cv2.rectangle(img, (160, 143), (180, 163), _COLOR_PRED_SPEECH, -1)

        cv2.putText(img, 'Pred. Noise:', (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    lineType=cv2.CV_AA)
        cv2.rectangle(img, (160, 173), (180, 193), _COLOR_PRED_NOISE, -1)

    cv2.putText(img, 'Time: %.3fs' % t, (10, h - _HMAP_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                lineType=cv2.CV_AA)
    cv2.putText(img, 'Output value (likelihood):', (10, h - _HMAP_HEIGHT + 20),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                lineType=cv2.CV_AA)

    cv2.putText(img, 'Top view:', (w - _X_EXPAND + 10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                lineType=cv2.CV_AA)

def _bgrtuple2rgbstr(bgr):
    b, g, r = bgr
    return '#%02x%02x%02x' % (r, g, b)

def main(root, outdir, method, sid, vmin, vmax, win_size, hop_size,
         min_sc, no_gt, add_sns, audio_onset, no_video, min_sns, method_name):
    # create output directory and sub-dir
    os.mkdir(outdir)
    odatadir = os.path.join(outdir, 'data')
    ofigdir = os.path.join(outdir, 'fig')
    os.mkdir(odatadir)
    os.mkdir(ofigdir)

    # load result
    rdir = os.path.join(root, 'results', method)

    # DOAs
    doa = np.load(os.path.join(rdir, 'doas.npy'))
    doa_azi = np.arctan2(doa[:,1], doa[:,0])
    index_doa_azi_sort = sorted(enumerate(doa_azi), key=lambda x: x[1])
    perm = [index for index, _ in index_doa_azi_sort]
    doa_azi_sort = [a for _, a in index_doa_azi_sort]
    doa_file = os.path.join(odatadir, 'doa')
    np.savetxt(doa_file, doa_azi_sort, fmt='%.5g')
    nlist = apkit.neighbor_list(doa, math.pi / 180 * 5)

    # project doa to image x-coordinate
    if not no_video:
        a2d, _ = cv2.projectPoints(doa * 2, _CAM_R, _CAM_T,
                                   _CAM_MATRIX, _CAM_DIST)
        a2dx = [int(x) + _X_EXPAND if abs(x) < 100000 and dx > 0.0 else None
                        for ((x, y), ), (dx, dy, dz) in zip(a2d, doa)]

    # heat values
    if not add_sns:
        phi = np.load(os.path.join(rdir, sid + '.npy'))
    else:
        phi, sns = evaluation.load_2tasks_heat(rdir, sid)

    ndoa, nframe = phi.shape
    assert ndoa == len(doa)
    heat_data = np.stack([phi, sns], axis=-1) if add_sns else phi
    for t in range(nframe):
        np.savetxt(os.path.join(odatadir, 'h%06d' % t),
                   heat_data[:,t][perm], fmt='%.5g')

    # find local maxima
    lmax = apkit.local_maxima(phi, nlist, th_phi=min_sc)

    # load ground truth
    if not no_gt:
        with open(os.path.join(root, 'data', _GT_PATTERN % (sid, win_size, hop_size)), 'r') as f:
            gt = pickle.load(f)
        fid2gt = dict(gt)

    # load audio (only to see the frame rate)
    wav_file = os.path.join(root, 'data', '%s.wav' % sid)
    fs, _ = apkit.load_wav(wav_file)
    fr = Fraction(fs, hop_size)

    # load video
    if not no_video:
        vdir = os.path.join(root, 'video_proc', sid)
        stamps = np.loadtxt(os.path.join(vdir, 'stamps'))

        for fid in range(nframe):
            t = float(fid * hop_size + win_size / 2) / fs + audio_onset
            vimg = cv2.imread(os.path.join(vdir,
                                          'r%06d.png' % qvfid(t, stamps)))
            h, w, _ = vimg.shape
            img = np.ones((h + _HMAP_HEIGHT, w + 2 * _X_EXPAND, 3),
                          dtype=vimg.dtype) * 255
            img[:h, _X_EXPAND:-_X_EXPAND] = vimg

            img = plot_grid(img, a2dx)

            if not no_gt and fid in fid2gt and len(fid2gt[fid]) > 0:
                if add_sns:
                    gdoa = np.asarray([loc for loc, stype, spkid in fid2gt[fid]])
                else:
                    gdoa = np.asarray([loc for loc, stype, spkid in fid2gt[fid]
                                                                 if stype == 1])

                if len(gdoa) > 0:
                    n2d, _ = cv2.projectPoints(gdoa, _CAM_R, _CAM_T,
                                               _CAM_MATRIX, _CAM_DIST)
                    n2d = np.asarray(n2d, dtype=int)
                    n2d[:,0,0] += _X_EXPAND
                    for p in n2d:
                        try:
                            cv2.circle(img, tuple(p[0]), 40, _COLOR_GT_SPEECH, 5, lineType=cv2.CV_AA)
                        except OverflowError:
                            pass

                    # save ground truth data
                    gdoa_azi = np.arctan2(gdoa[:,1], gdoa[:,0])
                    np.savetxt(os.path.join(odatadir, 'g%06d' % fid), gdoa_azi, fmt='%.5g')
                else:
                    np.savetxt(os.path.join(odatadir, 'g%06d' % fid), [-9], fmt='%.5g')

                if add_sns:
                    gndoa = np.asarray([loc for loc, stype, spkid in fid2gt[fid]
                                                                  if stype != 1])
                    if len(gndoa) > 0:
                        n2d, _ = cv2.projectPoints(gndoa, _CAM_R, _CAM_T,
                                                   _CAM_MATRIX, _CAM_DIST)
                        n2d = np.asarray(n2d, dtype=int)
                        n2d[:,0,0] += _X_EXPAND
                        for p in n2d:
                            try:
                                cv2.circle(img, tuple(p[0]), 40, _COLOR_GT_NOISE, 5, lineType=cv2.CV_AA)
                            except OverflowError:
                                pass

                        # save ground truth data
                        gndoa_azi = np.arctan2(gndoa[:,1], gndoa[:,0])
                        np.savetxt(os.path.join(odatadir, 'f%06d' % fid), gndoa_azi, fmt='%.5g')
                    else:
                        np.savetxt(os.path.join(odatadir, 'f%06d' % fid), [-9], fmt='%.5g')
            else:
                np.savetxt(os.path.join(odatadir, 'g%06d' % fid), [-9], fmt='%.5g')
                if add_sns:
                    np.savetxt(os.path.join(odatadir, 'f%06d' % fid), [-9], fmt='%.5g')

            # hmap in fov
            if add_sns:
                plot_hmap_fov(img[h:], a2dx, phi[:, fid], sns[:, fid])
            else:
                plot_hmap_fov(img[h:], a2dx, phi[:, fid])

            # plot prediction
            ol = img.copy()
            for pid in lmax[fid]:
                px = a2dx[pid]
                if px is not None:
                    if add_sns and sns[pid, fid] * phi[pid, fid] > min_sns:
                        pcolor = _COLOR_PRED_SPEECH
                    else:
                        pcolor = _COLOR_PRED_NOISE
                    cv2.rectangle(ol, (px-10, 0), (px+10, img.shape[1]), pcolor, -1)
            img = 0.6 * img + 0.4 * ol

            # save prediction
            if add_sns:
                pdoa_azi = doa_azi[[pid for pid in lmax[fid]
                                        if sns[pid, fid] * phi[pid, fid] > min_sns]]
                qdoa_azi = doa_azi[[pid for pid in lmax[fid]
                                        if sns[pid, fid] * phi[pid, fid] <= min_sns]]
            else:
                pdoa_azi = doa_azi[lmax[fid]]

            if len(pdoa_azi) > 0:
                np.savetxt(os.path.join(odatadir, 'p%06d' % fid), pdoa_azi, fmt='%.5g')
            else:
                np.savetxt(os.path.join(odatadir, 'p%06d' % fid), [-9], fmt='%.5g')

            if add_sns:
                if len(qdoa_azi) > 0:
                    np.savetxt(os.path.join(odatadir, 'q%06d' % fid), qdoa_azi, fmt='%.5g')
                else:
                    np.savetxt(os.path.join(odatadir, 'q%06d' % fid), [-9], fmt='%.5g')

            # plot legend
            plot_legend(img, t, sid, method_name, add_sns)

            cv2.imwrite(os.path.join(ofigdir, 'v%06d.png' % fid), img)
    else:
        # no video
        for fid in range(nframe):
            if not no_gt and fid in fid2gt and len(fid2gt[fid]) > 0:
                if add_sns:
                    gdoa = np.asarray([loc for loc, stype, spkid in fid2gt[fid]])
                else:
                    gdoa = np.asarray([loc for loc, stype, spkid in fid2gt[fid]
                                                                 if stype == 1])

                if len(gdoa) > 0:
                    # save ground truth data
                    gdoa_azi = np.arctan2(gdoa[:,1], gdoa[:,0])
                    np.savetxt(os.path.join(odatadir, 'g%06d' % fid), gdoa_azi, fmt='%.5g')
                else:
                    np.savetxt(os.path.join(odatadir, 'g%06d' % fid), [-9], fmt='%.5g')

                if add_sns:
                    gndoa = np.asarray([loc for loc, stype, spkid in fid2gt[fid]
                                                                  if stype != 1])
                    if len(gndoa) > 0:
                        # save ground truth data
                        gndoa_azi = np.arctan2(gndoa[:,1], gndoa[:,0])
                        np.savetxt(os.path.join(odatadir, 'f%06d' % fid), gndoa_azi, fmt='%.5g')
                    else:
                        np.savetxt(os.path.join(odatadir, 'f%06d' % fid), [-9], fmt='%.5g')
            else:
                np.savetxt(os.path.join(odatadir, 'g%06d' % fid), [-9], fmt='%.5g')
                if add_sns:
                    np.savetxt(os.path.join(odatadir, 'f%06d' % fid), [-9], fmt='%.5g')

            # save prediction
            if add_sns:
                pdoa_azi = doa_azi[[pid for pid in lmax[fid]
                                        if sns[pid, fid] * phi[pid, fid] > min_sns]]
                qdoa_azi = doa_azi[[pid for pid in lmax[fid]
                                        if sns[pid, fid] * phi[pid, fid] <= min_sns]]
            else:
                pdoa_azi = doa_azi[lmax[fid]]

            if len(pdoa_azi) > 0:
                np.savetxt(os.path.join(odatadir, 'p%06d' % fid), pdoa_azi, fmt='%.5g')
            else:
                np.savetxt(os.path.join(odatadir, 'p%06d' % fid), [-9], fmt='%.5g')

            if add_sns:
                if len(qdoa_azi) > 0:
                    np.savetxt(os.path.join(odatadir, 'q%06d' % fid), qdoa_azi, fmt='%.5g')
                else:
                    np.savetxt(os.path.join(odatadir, 'q%06d' % fid), [-9], fmt='%.5g')

    script_file = os.path.join(outdir, 'plot.gp')
    with open(script_file, 'w') as s:
        if not no_video:
            print('set terminal pngcairo size 240,320', file=s)
        else:
            print('set terminal pngcairo size 800,600', file=s)
        print('set polar', file=s)
        print('unset border', file=s)
        print('unset margin', file=s)
        print('set tics scale 0', file=s)
        print('unset xtics', file=s)
        print('unset ytics', file=s)
        print('set rtics ("" 0, "" 0.25, "" 0.5, "" 0.75, "" 1.0)', file=s)
        print('unset raxis', file=s)
        print('set trange [-2*pi:2*pi]', file=s)
        print('set grid polar pi/6', file=s)
        print('set size square', file=s)
        if not no_video:
            print('set key bm', file=s)
        else:
            print('set key bot rm', file=s)
            print('set xrange [-1.3:1.3]', file=s)
            print('set yrange [-1.3:1.3]', file=s)
        print('set label at 1.2,0 "right" center rotate by -90 tc rgb "gray"', file=s)
        print('set label at -1.2,0 "left" center rotate by 90 tc rgb "gray"', file=s)
        print('set label at 0,1.2 "front" center tc rgb "gray"', file=s)
        print('set label at 0,-1.2 "rear" center tc rgb "gray"', file=s)
        print('do for [ii=0:%d] {' % (nframe - 1), file=s)
        print('  data=sprintf("< paste %s %s/h%%06d", ii)' % (doa_file, odatadir), file=s)
        print('  gdata=sprintf("%s/g%%06d", ii)' % (odatadir), file=s)
        print('  pdata=sprintf("%s/p%%06d", ii)' % (odatadir), file=s)
        if add_sns:
            print('  fdata=sprintf("%s/f%%06d", ii)' % (odatadir), file=s)
            print('  qdata=sprintf("%s/q%%06d", ii)' % (odatadir), file=s)
        print('  set output sprintf("%s/t%%06d.png", ii)' % ofigdir, file=s)
        if no_video:
            print('  set title sprintf("Method %s; Time %%.2fs; Frame #%%06d", ii * %g, ii)' % (method_name, 1.0 * hop_size / fs), file=s)
        if add_sns:
            print('  plot 1.1 w l lw 2 lc rgb "gray" notitle,' \
                        ' data u ($1+0.5*pi):2 w l lc rgb "%s" lw 2 title "SSL Likelihood",' \
                        ' data u ($1+0.5*pi):3 w l lc rgb "%s" lw 1 title "SNS Likelihood",' \
                        ' gdata u ($1+0.5*pi):(1.05) pt 6 ps 3 lw 3 lc rgb "%s" title "GT. Speech",' \
                        ' fdata u ($1+0.5*pi):(1.05) pt 6 ps 3 lw 3 lc rgb "%s" title "GT. Noise",' \
                        ' pdata u ($1+0.5*pi):(1.05) pt 2 ps 3 lw 3 lc rgb "%s" title "Pred. Speech",' \
                        ' qdata u ($1+0.5*pi):(1.05) pt 2 ps 3 lw 3 lc rgb "%s" title "Pred. Noise"' \
                        % tuple([_bgrtuple2rgbstr(c) for c in [_COLOR_OUTPUT_SSL, _COLOR_OUTPUT_SNS, _COLOR_GT_SPEECH, _COLOR_GT_NOISE, _COLOR_PRED_SPEECH, _COLOR_PRED_NOISE]]), file=s)
        else:
            print('  plot 1.1 w l lw 2 lc rgb "gray" notitle, data u ($1+0.5*pi):2 w l lc rgb "blue" lw 2 title "output value", gdata u ($1+0.5*pi):(1.05) pt 6 ps 3 lw 3 lc rgb "red" title "ground truth", pdata u ($1+0.5*pi):(1.05) pt 2 ps 3 lw 3 lc rgb "green" title "prediction"', file=s)
        print('}', file=s)
    audio_temp = os.path.join(outdir, '%s.wav' % sid)
    print('data and script generated, now run')
    print('  gnuplot %s && \\' % script_file)
    if not no_video:
        print('  for x in %s/v*.png; do z=${x##*/}; y=${x%%/*}/${z/v/t}; o=${x%%/*}/${z/v/m}; convert -page +0+0 ${x} -page +880+100 ${y} -flatten ${o}; done && \\' % ofigdir) 
    print('  gst-launch-1.0 filesrc location="%s" ! decodebin ! audioresample ! "audio/x-raw,rate=16000" ! deinterleave name=d' \
          '    interleave name=i ! audioconvert ! wavenc ! filesink location="%s"' \
          '    d.src_0 ! queue ! i.sink_0' \
          '    d.src_1 ! queue ! i.sink_1 && \\' % (wav_file, audio_temp))
    print('  gst-launch-1.0 multifilesrc location="%s/%s%%06d.png" ' \
          '    caps="image/png,framerate=%d/%d,pixel-aspect-ratio=1/1" ' \
          '    ! decodebin ! videorate ! videoconvert ! theoraenc ! oggmux name=mux ! filesink location=%s/%s.ogv ' \
          '    filesrc location="%s" ! decodebin ! audioconvert ! vorbisenc ! mux. ' \
                % (ofigdir, 'm' if not no_video else 't', fr.numerator, fr.denominator, outdir, sid, audio_temp))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate demo video')
    parser.add_argument('test', metavar='TEST_PATH', type=str,
                        help='path to test data and feature')
    parser.add_argument('outdir', metavar='OUTPUT_DIR', type=str,
                        help='output directory')
    parser.add_argument('-m', '--method', metavar='METHOD', type=str,
                        required=True, help='method id')
    parser.add_argument('-n', '--method-name', metavar='NAME', type=str,
                        required=True, help='method name')
    parser.add_argument('-s', '--sid', metavar='SID', type=str,
                        required=True, help='Segment ID')
    parser.add_argument('-v', '--vrange', metavar=('VMIN', 'VMAX'), type=float,
                        nargs=2, default=(0., 1.), help='(default 0:1) value range')
    parser.add_argument('-w', '--window-size', metavar='WIN_SIZE',
                        type=int, default=2048,
                        help='(default 2048) analysis window size')
    parser.add_argument('-o', '--hop-size', metavar='HOP_SIZE', type=int,
                        default=1024, help='(default 1024) hop size, '
                        'number of samples between windows')
    parser.add_argument('--min-score', metavar='SCORE', type=float,
                        default=0.0, help='(default 0.0) minimun score for peaks')
    parser.add_argument('--no-gt', action='store_true',
                        help='Do not load ground truth')
    parser.add_argument('--sns', action='store_true',
                        help='Results include speech/non-speech classificatoin')
    parser.add_argument('--audio-onset', metavar='AUDIO_ONSET', type=float,
                        default=0.0, help='(default 0.0) the audio onset timestamp in seconds')
    parser.add_argument('--no-video', action='store_true', help='no video')
    parser.add_argument('--min-sns', metavar='SNS_SCORE', type=float,
                        default=0.5, help='(default 0.5) minimun score for classify as speech')
    args = parser.parse_args()
    main(args.test, args.outdir, args.method, args.sid, args.vrange[0],
         args.vrange[1], args.window_size, args.hop_size, args.min_score,
         args.no_gt, args.sns, args.audio_onset, args.no_video, args.min_sns,
         args.method_name)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

