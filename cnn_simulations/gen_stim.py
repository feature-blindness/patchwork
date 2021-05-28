import numpy
import copy
from PIL import Image, ImageDraw
import os
import errno
import random
from operator import add
from palettable.tableau import Tableau_20
from palettable.cartocolors.qualitative import Prism_10, Safe_10, Vivid_10


# each shape consists of a number of segments
# each line is two tuples (startx, starty) & (endx endy)
def gen_shape1(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Shape 1: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

                            o (5)
        (0) o               o
            o [0] (6)       o [2]
            o     o   [1]   o
      (2) o o o o o o o o o o o o(3)
        (1) o     o         o (4)
                  o
            (8) o o o o o o (9)
                  o    [4]
              [3] o
                  o (7)

    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 10  # Number of vertices (see comment above)
    num_edge = 5  # Number of edges

    minmax_lens = [(5, 10), (10, 15), (5, 10), (6,11), (5,10)]  # (min, max) length of each edge
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
    else:
        lens = [seg_len_short, seg_len_long, seg_len_short, seg_len_short, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0]
    xx[2] = xx[1] - numpy.random.randint(0,3)
    xx[3] = xx[2] + lens[1]
    xx[4] = xx[3] - numpy.random.randint(0,2)
    xx[5] = xx[4]
    xx[6] = numpy.random.randint(xx[0]+2, xx[5]-2)
    xx[7] = xx[6]
    xx[8] = max(xx[6] - numpy.random.randint(0,2), xx[1]+2) # make sure there's space b/w [0] & [4]
    xx[9] = xx[8] + lens[4]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0] + lens[0]
    yy[2] = yy[1] - numpy.random.randint(1,4)
    yy[3] = yy[2]
    yy[4] = yy[3] + numpy.random.randint(0,2)
    yy[5] = yy[4] - lens[2]
    yy[6] = yy[2] - numpy.random.randint(0,2)
    yy[7] = yy[6] + lens[3]
    if yy[4] + 2 < yy[7]-1:
        yy[8] = numpy.random.randint(yy[4]+2, yy[7]-1)
    else:
        yy[8] = numpy.random.randint(yy[4] + 1, yy[7])
    yy[9] = yy[8]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[2], yy[2], xx[3], yy[3]),
                (xx[4], yy[4], xx[5], yy[5]),
                (xx[6], yy[6], xx[7], yy[7]),
                (xx[8], yy[8], xx[9], yy[9])]

    return segments


def gen_shape2(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Shape 2: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

                       o (3)                     o (7)
                       o               o (5)     o
         (0) o     [2] o           [3] o         o [4]
             o         o               o         o
         (2) o o o o o o o o o o o o o o o o o o o (9)
             o   [1]   o               o         o
         [0] o         o (4)           o         o
             o                         o (6)     o
         (1) o                                   o (8)

    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 10  # Number of vertices (see comment above)
    num_edge = 5  # Number of edges

    minmax_lens = [(5, 10), (10, 15), (5, 10), (5, 10), (5, 10)]  # (min, max) length of each edge
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
    else:
        lens = [seg_len_short, seg_len_long, seg_len_short, seg_len_short, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0]
    xx[2] = xx[0]
    xx[3] = numpy.random.randint(xx[2]+2, int(xx[2]+(lens[1]/2))-1)  # somewhere on first half of [1]
    xx[4] = xx[3]
    xx[5] = numpy.random.randint(int(xx[2]+lens[1]/2)+1, xx[2]+lens[1]-2)  # somewhere on second half
    xx[6] = xx[5]
    xx[7] = xx[2] + lens[1]
    xx[8] = xx[7]
    xx[9] = xx[2] + lens[1]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0] + lens[0]
    yy[2] = numpy.random.randint(yy[0]+2, yy[1]-2)
    yy[3] = numpy.random.randint(yy[2]-lens[2]+2, yy[2]-2)
    yy[4] = yy[3] + lens[2]
    yy[5] = numpy.random.randint(yy[2]-lens[3]+2, yy[2]-2)
    yy[6] = yy[5] + lens[3]
    yy[7] = numpy.random.randint(yy[2]-lens[4]+2, yy[2]-2)
    yy[8] = yy[7] + lens[4]
    yy[9] = yy[2]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[2], yy[2], xx[9], yy[9]),
                (xx[3], yy[3], xx[4], yy[4]),
                (xx[5], yy[5], xx[6], yy[6]),
                (xx[7], yy[7], xx[8], yy[8])]

    return segments


def gen_shape3_not_used():
    '''
    NOT USING THIS AS NUM SEGMENTS DOES NOT MATCH OTHERS
    '''
    '''
    Shape 3: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

                 [0]
         (0) o o o o o o (2)
             o         o
         [1] o         o [2]                     o (7)
             o         o          (4)            o
         (1) o o o o o o o o o o o o o o o o o o o (8)
                [3]    o           o             o
                      (3)      [4] o             o [6]
                                   o             o
                               (5) o o o o o o o o (6)
                                          [5]

    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 9  # Number of vertices (see comment above)
    num_edge = 7  # Number of edges

    minmax_lens = [(4, 8), (3, 6), (3, 7), (11, 16), (3, 6), (4, 8), (4, 7)]  # (min, max) lengths
    lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
            for ee in range(num_edge)]  # Randomly chosen edge lengths
    # lens[3] = lens[2]  # Tie [3] and [2]
    # lens[7] = lens[5]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0]
    xx[2] = xx[0] + lens[0]
    xx[3] = xx[2]
    xx[4] = xx[1] + lens[3] - lens[5]
    xx[5] = xx[4]
    xx[6] = xx[1] + lens[3]
    xx[7] = xx[6]
    xx[8] = xx[1] + lens[3]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0] + lens[1]
    yy[2] = yy[0]
    yy[3] = yy[2] + lens[2]
    yy[4] = yy[1]
    yy[5] = yy[4] + lens[4]
    yy[6] = yy[5]
    yy[7] = yy[6] - lens[6]
    yy[8] = yy[1]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[0], yy[0], xx[2], yy[2]),
                (xx[1], yy[1], xx[8], yy[8]),
                (xx[2], yy[2], xx[3], yy[3]),
                (xx[4], yy[4], xx[5], yy[5]),
                (xx[5], yy[5], xx[6], yy[6]),
                (xx[6], yy[6], xx[7], yy[7])]

    return segments


def gen_shape3(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Shape 3: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

                           o (2)
                           o             o (5)
                     [3]   o             o
             (4) o o o o o o o o o o o o o o o o o (7)
                 o         o             o
             [2] o         o [1]         o [4]
                 o         o             o
         (0) o o o o o o o o (1)         o (6)
                 o   [0]
             (3) o

    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 8  # Number of vertices (see comment above)
    num_edge = 5  # Number of edges

    minmax_lens = [(5, 10), (5, 10), (5, 10), (12, 17), (5, 10)]  # (min, max) length of each edge
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
    else:
        lens = [seg_len_short, seg_len_short, seg_len_short, seg_len_long+2, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0] + lens[0]
    xx[2] = xx[1]
    xx[3] = numpy.random.randint(xx[0]+1, xx[1]-2)  # somewhere on first half of [1]
    xx[4] = xx[3]
    xx[5] = numpy.random.randint(xx[2]+2, xx[4]+lens[3]-1)  # somewhere on second half
    xx[6] = xx[5]
    xx[7] = xx[4] + lens[3]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0]
    yy[2] = yy[1] - lens[1]
    yy[3] = yy[0] + numpy.random.randint(0, int(lens[2]/2))
    yy[4] = yy[3] - lens[2]
    yy[5] = yy[4] - numpy.random.randint(1, int(lens[4]/2))
    yy[6] = yy[5] + lens[4]
    yy[7] = yy[4]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[1], yy[1], xx[2], yy[2]),
                (xx[3], yy[3], xx[4], yy[4]),
                (xx[4], yy[4], xx[7], yy[7]),
                (xx[5], yy[5], xx[6], yy[6])]

    return segments


def gen_shape4(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Shape 6: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

                (0) o                         o (6)
                    o          [1]            o
          (2) o o o o o o o o o o o o o o o o o o o (3)
                    o                         o
                    o [0]                 [4] o
            [2]     o                         o      [3]
    (5) o o o o o o o o (4)             (8) o o o o o o o o o (9)
                    o                         o
                (1) o                         o (7)

    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 10  # Number of vertices (see comment above)
    num_edge = 5  # Number of edges

    minmax_lens = [(5, 10), (10, 15), (5, 10), (5, 10), (5, 10)]  # (min, max) length of each edge
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
    else:
        lens = [seg_len_short, seg_len_long, seg_len_short, seg_len_short, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0]
    xx[2] = numpy.random.randint(xx[1]-2, xx[0])
    xx[3] = xx[2] + lens[1]
    xx[4] = numpy.random.randint(xx[0], xx[0]+3)
    xx[5] = xx[4] - lens[2]
    xx[6] = numpy.random.randint(xx[3]-2, xx[3])
    xx[7] = xx[6]
    xx[8] = numpy.random.randint(xx[6]-3, xx[6])
    xx[9] = xx[8] + lens[3]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0] + lens[0]
    yy[2] = numpy.random.randint(yy[0], yy[0]+1)
    yy[3] = yy[2]
    yy[4] = numpy.random.randint(yy[2]+2, yy[1]-2)
    yy[5] = yy[4]
    yy[6] = yy[0]
    yy[7] = yy[6] + lens[4]
    yy[8] = yy[4]
    yy[9] = yy[8]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[2], yy[2], xx[3], yy[3]),
                (xx[4], yy[4], xx[5], yy[5]),
                (xx[6], yy[6], xx[7], yy[7]),
                (xx[8], yy[8], xx[9], yy[9])]

    return segments


def gen_shape5(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Shape 5: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

         (0) o
             o         [1]
         (4) o o o o o o o o o o o o o (5)
             o                       o
             o [0]               [4] o
             o         [2]           o
     (2) o o o o o o o o o o o o o o o o o o (3)
             o                       o
             o                       o
             o         [3]           o
         (1) o o o o o o o o o o o o o (6)
                                     o
                                     o (7)


    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 8  # Number of vertices (see comment above)
    num_edge = 5  # Number of edges

    minmax_lens = [(6, 11), (5, 10), (11, 18), (5, 10), (6, 11)]  # (min, max) length of each edge
    lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
            for ee in range(num_edge)]  # Randomly chosen edge lengths
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
        lens[3] = numpy.random.randint(lens[1], 10)
        lens[4] = numpy.random.randint(lens[0], lens[0] + 2)
    else:
        lens = [seg_len_short, seg_len_short, seg_len_long, seg_len_short, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0]
    xx[2] = numpy.random.randint(xx[0]-2, xx[0]+1)
    xx[3] = xx[2] + lens[2]
    xx[4] = xx[1]
    xx[5] = xx[4] + lens[1]
    xx[6] = xx[1] + lens[3]
    xx[7] = xx[5]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0] + lens[0]
    if yy[0] + 3 < yy[1] - 2:
        yy[2] = numpy.random.randint(yy[0]+3, yy[1]-2)
    else:
        yy[2] = numpy.random.randint(yy[0] + 2, yy[1] - 2)
    yy[3] = yy[2]
    yy[4] = numpy.random.randint(yy[0], yy[2]-1)
    yy[5] = yy[4]
    yy[6] = yy[1]
    yy[7] = yy[5] + lens[4]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[1], yy[1], xx[6], yy[6]),
                (xx[2], yy[2], xx[3], yy[3]),
                (xx[4], yy[4], xx[5], yy[5]),
                (xx[5], yy[5], xx[7], yy[7])]

    return segments


def gen_shape6(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Shape 4: Numbers in parentheses indicate vertex number
             Numbers in square brackets indicate edge number

         (0) o
             o  [0]                  o (7)
             o                       o
             o     [1]               o
         (2) o o o o o o o (3)       o [4]
             o                       o
             o                       o
             o          (4)          o
         (1) o o o o o o o o o o o o o (6)
                 [2]     o
                         o [3]
                         o
                         o (5)


    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_vert = 8  # Number of vertices (see comment above)
    num_edge = 5  # Number of edges

    minmax_lens = [(5, 10), (5, 10), (10, 15), (5, 10), (5, 10)]  # (min, max) length of each edge
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
    else:
        lens = [seg_len_short, seg_len_short, seg_len_long, seg_len_short, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    xx[1] = xx[0]
    xx[2] = xx[0]
    xx[3] = xx[2] + lens[1]
    xx[4] = numpy.random.randint(xx[1]+2, xx[1]+lens[2]-2)
    xx[5] = xx[4]
    xx[6] = xx[1] + lens[2]
    xx[7] = xx[6]

    yy = [0 for ii in range(num_vert)]  # All y-coords
    yy[1] = yy[0] + lens[0]
    yy[2] = numpy.random.randint(yy[0]+2, yy[1]-2)
    yy[3] = yy[2]
    yy[4] = yy[1]
    yy[5] = yy[4] + lens[3]
    yy[6] = yy[1]
    yy[7] = yy[6] - lens[4]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[1], yy[1], xx[6], yy[6]),
                (xx[2], yy[2], xx[3], yy[3]),
                (xx[4], yy[4], xx[5], yy[5]),
                (xx[6], yy[6], xx[7], yy[7])]

    return segments


def gen_rand_shape(seg_len_cue=False, seg_len_short=6, seg_len_long=11):
    '''
    Random set of five segments

    :returns    list of segments, where each segment is a 4-tuple: (x0,y0,x1,y1)
    '''

    num_edge = 5  # Number of edges
    num_vert = 10  # Number of vertices
    canvas = (26,26) # Size of canvas
    min_dist = 2 # minimum separation between segments along x & y dimensions

    minmax_lens = [(5, 10), (5, 10), (10, 15), (5, 10), (5, 10)]  # (min, max) length of each edge
    if seg_len_cue == False:
        lens = [numpy.random.randint(minmax_lens[ee][0], minmax_lens[ee][1])
                for ee in range(num_edge)]  # Randomly chosen edge lengths
    else:
        lens = [seg_len_short, seg_len_short, seg_len_long, seg_len_short, seg_len_short]

    xx = [0 for ii in range(num_vert)]  # All x-coords
    yy = [0 for ii in range(num_vert)]  # All y-coords
    used_xx = []
    used_yy = []
    for ee in range(num_edge):
        vert = False # whether edge is vertial
        if numpy.random.rand() <= 0.5:
            vert = True
        vv0 = ee * 2 # first vertex of segment
        vv1 = vv0 + 1 # second vertex
        minx = 1
        maxx = canvas[0] - 11
        miny = 1
        maxy = canvas[1] - 11

        xx[vv0] = numpy.random.randint(minx, maxx)
        if used_xx != []:
            while (min([abs(xx[vv0]-xi) for xi in used_xx]) < min_dist):  # Ensure segs are min_dist apart
                xx[vv0] = numpy.random.randint(minx, maxx)
        used_xx.append(xx[vv0])

        if vert is True:
            xx[vv1] = xx[vv0]
        else:
            xx[vv1] = xx[vv0] + lens[ee]

        yy[vv0] = numpy.random.randint(miny, maxy)
        if used_yy != []:
            while (min([abs(yy[vv0]-yi) for yi in used_yy]) < min_dist):  # Ensure segs are min_dist apart
                yy[vv0] = numpy.random.randint(miny, maxy)
        used_yy.append(yy[vv0])

        if vert is True:
            yy[vv1] = yy[vv0] + lens[ee]
        else:
            yy[vv1] = yy[vv0]

    ### If any index is negative, shift the shape
    if min(xx) < 0:
        xx = [ii-min(xx) for ii in xx]
    if min(yy) < 0:
        yy = [ii-min(yy) for ii in yy]

    ### Define the shape based on the vertex and edge labels in the comment above
    ### Shape is a list of segments, where each is a tuple (x0, y0, x1, y1)
    segments = [(xx[0], yy[0], xx[1], yy[1]),
                (xx[2], yy[2], xx[3], yy[3]),
                (xx[4], yy[4], xx[5], yy[5]),
                (xx[6], yy[6], xx[7], yy[7]),
                (xx[8], yy[8], xx[9], yy[9])]

    return segments


def draw_shape(segments, cat_col=(0,0,0), cat_loc=(250,250), shift_type='random',
               orientation=0, flip_lr=False, flip_tb=False, cell_size=4,
               imsize=(500,500), all_cols=[], unif_col=False, unif_seg = True, max_col=256,
               all_same=False, im_patches=None, col_del=50, shape_freq=None, avail_cols=[(100,100,100)],
               jitter_size=0, shift_centre=True):
    '''
    :parameter  segments: list of 4-tuples (x0, y0, x1, y1)
    '''

    size_imx = imsize[0]
    size_imy = imsize[1]
    im = Image.new('RGB', (size_imx, size_imy), color='white')
    drawing = ImageDraw.Draw(im)

    drawing_patches = ImageDraw.Draw(im_patches)

    ### Find (x,y) coordinates of each rectangle in each segment
    drawing_coords = [] # Coordinates of all the rectangles within all the segments
    num_cells_all = [] # list of number of cells in each segment
    for si in segments:
        x0 = si[0]
        y0 = si[1]
        x1 = si[2]
        y1 = si[3]

        ### Edges are only vertical or horizontal, work out which one
        jsz =  jitter_size # 0.3  # jitter size
        vertical = False
        if x0 == x1:
            vertical = True
        if vertical is True:
            num_cells = abs(y1 - y0)
            # num_cells = abs(si[3] - si[1])  # Number of cells in edge
            xx0 = [x0+numpy.random.uniform(-jsz,jsz) for ii in range(num_cells)] # x0 coordinates for all cells in segment are same
            # xx1 = [x0+1 for ii in range(num_cells)] # x1 coordinates also same
            xx1 = [xi+1+numpy.random.uniform(-jsz, jsz) for xi in xx0]
            yy0 = list(numpy.linspace(y0, y1, num_cells+1))  # y0 coordinates form a sequence
            yy0 = [int(ii) for ii in yy0[:-1]]  # float to integer; discard last index
            yy0 = [yi+numpy.random.uniform(-jsz, jsz) for yi in yy0] # jitter
            yy1 = [ii+1+numpy.random.uniform(-jsz, jsz) for ii in yy0]  # 1 cell width
        else: # Horizontal segment
            num_cells = abs(x1 - x0)
            # num_cells = abs(si[2] - si[0])
            xx0 = list(numpy.linspace(x0, x1, num_cells+1))  # x0 coordinates form a sequence
            xx0 = [int(ii) for ii in xx0[:-1]]  # float to int; discard last index, which was extra
            xx0 = [xi+numpy.random.uniform(-jsz, jsz) for xi in xx0]
            xx1 = [ii+1+numpy.random.uniform(-jsz, jsz) for ii in xx0]  # 1+jsz width
            yy0 = [y0+numpy.random.uniform(-jsz, jsz) for ii in range(num_cells)]  # y0 coords for all cells in segment are same
            yy1 = [y0+1+numpy.random.uniform(-jsz, jsz) for ii in range(num_cells)]  # y1 coordinates also same

        xx0 = [ii*cell_size for ii in xx0]
        xx1 = [ii*cell_size for ii in xx1]
        yy0 = [ii*cell_size for ii in yy0]
        yy1 = [ii*cell_size for ii in yy1]

        ### Convert to int as these are pixel coordinates
        xx0 = [int(ii) for ii in xx0]
        xx1 = [int(ii) for ii in xx1]
        yy0 = [int(ii) for ii in yy0]
        yy1 = [int(ii) for ii in yy1]

        num_cells_all.append(num_cells)
        drawing_coords.append([xx0, xx1, yy0, yy1])

    ### Move to middle of the canvas
    ### we do this by matching the middle of shape to middle of canvas
    minx = 9999 # very large number as it maintains running count of smallest x
    maxx = 0
    miny = 9999
    maxy = 0
    for subseg in drawing_coords:
        ### Get the minimum & max coordinates to work out shape size
        minx_subseg = min(min(subseg[0]), min(subseg[1]))
        maxx_subseg = max(max(subseg[0]), max(subseg[1]))
        miny_subseg = min(min(subseg[2]), min(subseg[3]))
        maxy_subseg = max(max(subseg[2]), max(subseg[3]))
        if minx_subseg < minx:
            minx = minx_subseg
        if maxx_subseg > maxx:
            maxx = maxx_subseg
        if miny_subseg < miny:
            miny = miny_subseg
        if maxy_subseg > maxy:
            maxy = maxy_subseg
    shape_width = maxx - minx
    shape_height = maxy - miny
    shape_cx = int(shape_width / 2)  # centre of shape x; convert to int in case image_size is odd
    shape_cy = int(shape_height / 2)
    image_cx = int(imsize[0] / 2) # convert to int in case image_size is odd
    image_cy = int(imsize[1] / 2)
    shift_x = image_cx - shape_cx
    shift_y = image_cy - shape_cx

    ### Shift all coordinates by shift_x & shift_y
    for subseg in drawing_coords:
        subseg[0] = [ii+shift_x for ii in subseg[0]]
        subseg[1] = [ii+shift_x for ii in subseg[1]]
        subseg[2] = [ii+shift_y for ii in subseg[2]]
        subseg[3] = [ii+shift_y for ii in subseg[3]]

    ### Shift such that there is a cell in the middle of the image
    if shift_type == 'closest':  # choose the cell that will is closest to centre and shift to centre
        dist_xx = []  # Will contain x-distances of each cell to centre
        dist_yy = []
        dist_xy = []  # Will contain sum xx+yy
        mindist = 9999  # Will be updated to minimum distance, therefore init to large number
        minsid = 0  # Will contain the segment id that has shortest distance
        for sid, subseg in enumerate(drawing_coords):
            dxi = [image_cx - ii for ii in subseg[0]]
            dyi = [image_cy - ii for ii in subseg[2]]
            abs_dxi = [abs(ii) for ii in dxi]  # Use absolute distance to work out x+y dist
            abs_dyi = [abs(jj) for jj in dyi]
            dxy = list(map(add, abs_dxi, abs_dyi))
            minxy = min(dxy)
            if minxy < mindist:
                mindist = minxy
                minsid = sid
            dist_xx.append(dxi)
            dist_yy.append(dyi)
            dist_xy.append(dxy)

        ### Get index of minimum x+y and use it to find smallest x-dist & y-dist
        shift_sid = minsid
        shift_cellix = dist_xy[minsid].index(mindist)
        shift_dx = dist_xx[minsid][shift_cellix] - int(cell_size / 2)
        shift_dy = dist_yy[minsid][shift_cellix] - int(cell_size / 2)
    elif shift_type == 'random':  # randomly choose a cell in the shape that will be centered
        shift_sid = numpy.random.randint(len(segments))
        shift_cellix = numpy.random.randint(1,len(drawing_coords[shift_sid][0])-1)
        shift_dx = image_cx - drawing_coords[shift_sid][0][shift_cellix] - int(cell_size / 2)
        shift_dy = image_cx - drawing_coords[shift_sid][2][shift_cellix] - int(cell_size / 2)

    ### Shift all coordinates by shift_dx,dy so that chosen cell is at centre of image
    for subseg in drawing_coords:
        subseg[0] = [ii+shift_dx for ii in subseg[0]]
        subseg[1] = [ii+shift_dx for ii in subseg[1]]
        subseg[2] = [ii+shift_dy for ii in subseg[2]]
        subseg[3] = [ii+shift_dy for ii in subseg[3]]

    ### Draw each segment
    for sid, subseg in enumerate(drawing_coords):
        xx0 = subseg[0]
        xx1 = subseg[1]
        yy0 = subseg[2]
        yy1 = subseg[3]
        num_cells = num_cells_all[sid]
        if unif_seg is True:
            if use_palette is True:
                color_ix = numpy.random.randint(0, len(avail_cols))
                seg_color = avail_cols[color_ix]
            else:
                seg_color = (numpy.random.randint(max_col),
                         numpy.random.randint(max_col),
                         numpy.random.randint(max_col))
                seg_color = tuple([cc * int(256 / max_col) for cc in seg_color])
                while (seg_color in all_cols):  # Ensure that color of patch is not one of diganostic colors
                    seg_color = (numpy.random.randint(max_col),
                             numpy.random.randint(max_col),
                             numpy.random.randint(max_col))
                    seg_color = tuple([cc * int(256 / max_col) for cc in seg_color])  # rescale
        for ci in range(num_cells):  # For each cell
            if unif_col is False and unif_seg is False: # sample different color for each patch
                if all_same is True:
                    color = (100, 100, 100)
                else:
                    if use_palette is True:
                        color_ix = numpy.random.randint(0, len(avail_cols))
                        color = avail_cols[color_ix]
                    else:
                        color = (numpy.random.randint(max_col),
                                 numpy.random.randint(max_col),
                                 numpy.random.randint(max_col))
                        color = tuple([cc*int(256/max_col) for cc in color])
                        while (color in all_cols): # Ensure that color of patch is not one of diganostic colors
                            color = (numpy.random.randint(max_col),
                                     numpy.random.randint(max_col),
                                     numpy.random.randint(max_col))
                            color = tuple([cc * int(256 / max_col) for cc in color]) # rescale
            elif unif_col is True: # make color of each patch = cat_col
                color = cat_col
            elif unif_seg is True:
                color = seg_color

            outline = None
            if unif_seg == True:
                if sid == shift_sid:  # diagnostic segment
                    color = cat_col
                    # outline = 'black' ### DEBUG
            else: # whole segment same colour
                if ci == shift_cellix and sid == shift_sid:  # diagnostic patch(es)
                    color = None ### To prevent overlap below
                    # outline = 'black' ### DEBUG
            only_diag_patch = False  # draw only the diagnostic patch and nothing else -- for Debug
            if only_diag_patch is not True:
                drawing.rectangle([(xx0[ci], yy0[ci]), (xx1[ci], yy1[ci])], outline=outline, fill=color)

    ### If only a single patch, make sure that it does not get hidden above
    if unif_seg is False: # single patch that is diagnostic
        ### Make the central pixel category correlated (it will be moved to category location later)
        cxx0 = drawing_coords[shift_sid][0][shift_cellix]
        cxx1 = cxx0 + cell_size # make sure central patch is always cell_size (not too small due to jitter)
        # cxx1 = drawing_coords[shift_sid][1][shift_cellix]
        cyy0 = drawing_coords[shift_sid][2][shift_cellix]
        cyy1 = cyy0 + cell_size # make sure central patch is always cell_size (not too small due to jitter)
        # cyy1 = drawing_coords[shift_sid][3][shift_cellix]
        drawing.rectangle([(cxx0, cyy0), (cxx1, cyy1)], outline=None, fill=cat_col)

    ### Rotate image according to given orientation
    im = im.rotate(orientation)

    ### Check whether to flip
    if flip_lr == True:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_tb == True:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)


    if (cat_loc is None):
        minmax_locs = (200, 400)  # Min & max x (and y) values for fixed cell
        cat_loc = (numpy.random.randint(minmax_locs[0], minmax_locs[1]),
                   numpy.random.randint(minmax_locs[0], minmax_locs[1]))

    ### Shift central cell to the location based on category
    if shift_centre == True:
        cat_shift_x = cat_loc[0] - image_cx
        cat_shift_y = cat_loc[1] - image_cy
        im = im.transform(im.size, Image.AFFINE,(1,0,-cat_shift_x,0,1,-cat_shift_y),
                          fillcolor=(255,255,255))
        # DEBUG START - check overlap in patches of diff categories
        if 'cxx0' in locals(): # Check 'cxx0' exists, i.e. cat_col is not None
            drawing_patches.rectangle([(cxx0+cat_shift_x, cyy0+cat_shift_y),
                                       (cxx1+cat_shift_x, cyy1+cat_shift_y)], outline='black', fill=cat_col)
        # DEBUG END - check overlap in patches of diff categories

    return (im, im_patches)


imsize = (600, 600)
nshapes = 5
nseed = 10 # 10 each seed has a different stimuli set
ntrain = 2000  # 2000
ntest = 500  # 500
exp5 = False # In Exp5 train only on non-shape -- i.e. pinv=1
cell_size = 12  # Size of each cell (patch) in Elmer shape
upright = False # are all shapes upright; used for rotation net
min_cell_size = 6 # Only used if cell_size is category-correlated (dcell_size)
max_cell_size = min_cell_size + (2*nshapes)
min_ncells_short = 6 # Used if num cells in segment is category-correlated (dnum_cells)
delta_ncells = 2 # difference in num_cells per segment amongst two shapes
max_ncells_short = min_ncells_short + (delta_ncells * nshapes)
min_ncells_long = 11 # Used if num cells in segment is category-correlated (dnum_cells)
max_ncells_long = min_ncells_long + (delta_ncells * nshapes)
if exp5 == True:
    pinv_list  = [1] # list of proportion of trials where shape-cue is invalid
else:
    pinv_list  = [0, 0.2] # list of proportion of trials where shape-cue is invalid
# pinv_list  = [0] # list of proportion of trials where shape-cue is invalid
rand_invalid = True # indicates whether invalid trials consist of random shape or one of other 4 shapes
rand_id = -1 # an unused shape_id number to indicate draw random shape
jit = 0.3 #0.3 # size of jitter; jit=0.5 will mean consec cells could be up to 1 cell-width displaced
orientations = numpy.array([0, 90, 180, 270])
max_col_del = 100 # Range of random colours added or substracted from central strip
shift_type = 'random'  # 'closest' / 'random' How to shift shape s:t there is cell at the centre
minmax_locs = (200, 400)  # Min & max x (and y) values for fixed cell
if exp5 == True:
    conditions = ['training', 'test_same', 'test_cv'] # 
else:
    conditions = ['training', 'test_same', 'test_diff', 'test_nodiag', 'test_inv', 'test_cv']
experiment = 'dcol_invalid' # 'type of experiment
'''
              dpatch_invalid: same as dpatch, but shape is diagnostic only for (1-pinvalid) trials
              dcol_invalid: each category has a unique colour that occurs only for that category
              dcell_size_invalid: the size of each cell is category-correlated
              dunicol: the color of the entire figure is diagnostic
'''
# if experiment is 'dcol_seg_noshape' or experiment is 'dcol_seg':
if experiment == 'dcol_seg_noshape' or experiment == 'dcol_seg' or experiment == 'dcol_invalid':
    unif_seg = True
else:
    unif_seg = False

if experiment == 'dcol_seg_noshape':
    shift_type = 'closest'

if experiment == 'dunicol':
    unif_col = True  # Indicates whether all patches are same colour (used in dunicol experiment)
else:
    unif_col = False

if experiment == 'dcol2':
    max_col = 2
else:
    max_col = 256

if experiment == 'dcol_same':
    all_same = True
else:
    all_same = False

if experiment == 'dnum_cells' or experiment == 'dnum_cells_invalid':
    seg_len_cue = True
    shift_centre = False
else:
    seg_len_cue = False
    shift_centre = True

min_freq = 0.2 # minimum frequency of sinusoidal grating (for 'dfreq')
max_freq = 1 # max frequency of sinusoidal grating (for 'dfreq')

### Define some maximally distinct colour palettes
use_palette = True  # otherwise uses random samples from [0,255] for each of r,g,b
### Kelly's palette (from Kelly, 1965)
# pal = [(255, 179, 0), (128, 62, 117), (255, 104, 0), (166, 189, 215), (193, 0, 32),
#          (206, 162, 98), (129, 112, 102), (0, 125, 52), (246, 118, 142), (0, 83, 138),
#          (255, 122, 92), (83, 55, 122), (255, 142, 0), (179, 40, 81), (244, 200, 0),
#          (127, 24, 13), (147, 170, 0), (89, 51, 21), (241, 58, 19), (35, 44, 22)]
### Brewer: Tableau
pal = Tableau_20.colors
pal = [(ii[0], ii[1], ii[2]) for ii in pal]
### Brewer: Safe + Vivid
# pal = Safe_10.colors + Vivid_10.colors[0:1] + [Vivid_10.colors[3]] + [Vivid_10.colors[7]] + [Vivid_10.colors[9]]  # Palette from Brewer
# pal = [(ii[0], ii[1], ii[2]) for ii in pal]
### Grayscale
# pal = [(ii, ii, ii) for ii in range(0,255,10)]

for pinvalid in pinv_list:
    for seed in range(nseed):
        cat_colors = []  # Will contain tuples indicating colors for each shape
        cat_locs = []  # Will contain tuples indicating location of fixed cell for each shape
        cat_sizes = []  # List of cell_sizes for each category
        cat_ncells = []  # List of number of cells for each category
        col_deltas = []  # This list will contain random delta colours for the diagnostic patch
        random.shuffle(pal)  # Shuffle pal as first few will be used as diagnostic cols
        avail_cols = pal[nshapes:]

        # Location used for all categories if experiment == 'dcol_oneloc_invalid'
        all_shape_loc = (numpy.random.randint(minmax_locs[0], minmax_locs[1]),
                         numpy.random.randint(minmax_locs[0], minmax_locs[1]))

        ### Generate all shapes and store in subdirectories
        for jj in range(nshapes):
            if use_palette is True:
                shape_color = pal[jj]
            else: # randomly sample from [0,255] each of r,g,b
                shape_color = (numpy.random.randint(max_col),
                               numpy.random.randint(max_col),
                               numpy.random.randint(max_col))
                shape_color = tuple([cc * int(256 / max_col) for cc in shape_color])
                while (shape_color in cat_colors):  # Ensure that shape_color is unique
                    shape_color = (numpy.random.randint(max_col),
                                   numpy.random.randint(max_col),
                                   numpy.random.randint(max_col))
                    shape_color = tuple([cc * int(256 / max_col) for cc in shape_color]) # rescale
            if experiment != 'dcol_oneloc_invalid':
                shape_loc = (numpy.random.randint(minmax_locs[0], minmax_locs[1]),
                             numpy.random.randint(minmax_locs[0], minmax_locs[1]))
                while (shape_loc in cat_locs):  # Ensure that shape_loc is unique
                    shape_loc = (numpy.random.randint(minmax_locs[0], minmax_locs[1]),
                                 numpy.random.randint(minmax_locs[0], minmax_locs[1]))
            else:
                shape_loc = all_shape_loc
            delcol = numpy.random.randint(max_col_del)
            cat_colors.append(shape_color)
            cat_locs.append(shape_loc)
            col_deltas.append(delcol)

        cat_freqs = numpy.linspace(min_freq, max_freq, nshapes)
        cat_freqs = numpy.random.permutation(cat_freqs) # shuffle
        cat_sizes = numpy.linspace(min_cell_size, max_cell_size, nshapes)
        cat_sizes = numpy.random.permutation(cat_sizes) # shuffle
        short_seg_list = list(range(min_ncells_short, max_ncells_short, delta_ncells))
        long_seg_list = list(range(min_ncells_long, max_ncells_long, delta_ncells))
        cat_ncells = [short_seg_list, long_seg_list]
        cat_ncells = list(zip(*cat_ncells)) # list of tuples
        cat_ncells = numpy.random.permutation(cat_ncells) # shuffle

        for set_type in iter(conditions):
            set_colors = [None for ss in range(nshapes)]
            set_locs = [None for ss in range(nshapes)]
            set_freqs = [None for ss in range(nshapes)]
            set_sizes = [None for ss in range(nshapes)]
            set_ncells = copy.deepcopy(cat_ncells)
            exclude_locs = False  # Used in Diff condition: excludes locations of diag patch
            if set_type == 'training':
                nitems = ntrain
                im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit*10)),
                                         'seed_' + str(seed),
                                         'p' + str(int(pinvalid*100)), 'train')
                if experiment == 'dpatch' or experiment == 'dpatch_invalid':
                    set_colors = copy.deepcopy(cat_colors)
                    set_locs = copy.deepcopy(cat_locs)
                elif experiment == 'dcol' or experiment == 'dcol2' or \
                     experiment == 'dunicol' or experiment == 'dcol_same' or \
                     experiment == 'dcol_invalid' or experiment == 'dcol_seg' or \
                     experiment == 'dcol_patch_invalid' or \
                     experiment == 'dcol_oneloc_invalid' or \
                     experiment == 'dcol_seg_noshape':
                    set_colors = copy.deepcopy(cat_colors)
                    set_locs = copy.deepcopy(cat_locs)  # irrelevant here as sampled for each item later
                elif experiment == 'dfreq':
                    set_freqs = copy.deepcopy(cat_freqs)
                    set_locs = copy.deepcopy(cat_locs)  # irrelevant here as sampled for each item later
                elif experiment == 'dnone':
                    set_colors = [None for ss in range(nshapes)]
                    set_locs = [None for ss in range(nshapes)]
                elif experiment == 'dcell_size' or experiment == 'dcell_size_invalid' or \
                     experiment == 'dnum_cells' or experiment == 'dnum_cells_invalid':
                    set_colors = [None for ss in range(nshapes)]
                    set_locs = [None for ss in range(nshapes)]
                    set_sizes = copy.deepcopy(cat_sizes)
            elif set_type == 'test_same' or set_type == 'test_diff':
                nitems = ntest
                if set_type == 'test_diff':
                    im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit*10)),
                                             'seed_' + str(seed),
                                             'p' + str(int(pinvalid * 100)), 'test_diff')
                    map_file = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit * 10)),
                                            'seed_' + str(seed),
                                            'p' + str(int(pinvalid * 100)), 'diff_map.txt')
                    neworder = numpy.random.permutation(nshapes)
                    ### Make sure no label is in the same location
                    oldorder = numpy.arange(0, nshapes)
                    while (numpy.sum(neworder == oldorder) > 0): # if any label is in same loc
                        neworder = numpy.random.permutation(nshapes) # find a new permutation
                    print("\n")
                    fh = open(map_file, "w", encoding="utf8")
                    for ss in range(nshapes):
                        print("Mapping shape_" + str(ss+1) + " <-- " + "shape_" + str(neworder[ss]+1))
                        fh.write("\nMapping shape_" + str(ss+1) + " <-- " + "shape_" + str(neworder[ss]+1))
                        set_colors[ss] = cat_colors[neworder[ss]]
                        set_locs[ss] = cat_locs[neworder[ss]]
                        set_sizes[ss] = cat_sizes[neworder[ss]]
                        set_ncells[ss] = cat_ncells[neworder[ss]]
                        if experiment == 'dfreq':
                            set_freqs[ss] = cat_freqs[neworder[ss]]
                    if experiment != 'dcol_oneloc_invalid':
                        exclude_locs = True # exclude the dignostic locations from training
                    fh.close()
                else:
                    im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit*10)),
                                             'seed_' + str(seed),
                                             'p' + str(int(pinvalid * 100)), 'test_same')
                    set_colors = copy.deepcopy(cat_colors)
                    set_locs = copy.deepcopy(cat_locs)
                    set_sizes = copy.deepcopy(cat_sizes)
                    if experiment == 'dfreq':
                        set_freqs = copy.deepcopy(cat_freqs)
            elif set_type == 'test_nodiag':
                nitems = ntest
                im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit*10)),
                                         'seed_' + str(seed),
                                         'p' + str(int(pinvalid * 100)), 'test_nodiag')
                set_colors = [None for ss in range(nshapes)]
                set_locs = [None for ss in range(nshapes)]
                set_sizes = [max_cell_size+1 for ss in range(nshapes)]
                if experiment == 'dfreq':
                    set_freqs = [None for ss in range(nshapes)]
            elif set_type == 'test_inv': # Test the invariant feature & no shape
                nitems = ntest
                im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit * 10)),
                                         'seed_' + str(seed),
                                         'p' + str(int(pinvalid * 100)), 'test_inv')
                set_colors = copy.deepcopy(cat_colors)
                set_locs = copy.deepcopy(cat_locs)
                set_sizes = copy.deepcopy(cat_sizes)
                if experiment == 'dfreq':
                    set_freqs = copy.deepcopy(cat_freqs)
            elif set_type == 'test_loc':
                nitems = ntest
                im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit*10)),
                                         'seed_' + str(seed),
                                         'p' + str(int(pinvalid * 100)), 'test_loc')
                set_colors = [None for ss in range(nshapes)]
                set_locs = copy.deepcopy(cat_locs) # this is still used for working out trans in draw_shape
                set_sizes = copy.deepcopy(cat_sizes)
                if experiment == 'dfreq':
                    set_freqs = copy.deepcopy(cat_freqs)
            elif set_type == 'test_cv':
                nitems = ntest
                im_folder = os.path.join('data', 'stim_' + experiment + '_jit' + str(int(jit*10)),
                                         'seed_' + str(seed),
                                         'p' + str(int(pinvalid * 100)), 'test_cv')
                set_colors = copy.deepcopy(cat_colors)
                set_locs = copy.deepcopy(cat_locs)
                set_sizes = copy.deepcopy(cat_sizes)
                if experiment == 'dfreq':
                    set_freqs = copy.deepcopy(cat_freqs)
            for jj in range(nshapes):
                # DEBUG START - check overlap in patches of diff categories
                im_patches = Image.new('RGB', (imsize[0], imsize[1]), color='white')
                drawing_patches = ImageDraw.Draw(im_patches)
                # DEBUG END - check overlap in patches of diff categories
                shape_id = jj + 1
                shape_color = set_colors[jj]
                shape_loc = set_locs[jj]
                shape_size = set_sizes[jj]
                shape_ncells = set_ncells[jj]
                col_del = col_deltas[jj]
                shape_freq = set_freqs[jj]
                for ii in range(nitems):
                    ### If invalid trial, change shape_id
                    if (experiment == 'dpatch_invalid' or experiment == 'dcol_invalid' or experiment == 'dcol_patch_invalid' or \
                        experiment == 'dcol_oneloc_invalid' or experiment == 'dcell_size_invalid' or \
                        experiment == 'dnum_cells_invalid' or experiment=='dunicol') and \
                       (pinvalid >= numpy.random.uniform(0, 1) and set_type == 'training'):  # some invalid trials during training
                        if rand_invalid == False:
                            all_shape_ids = [ii+1 for ii in range(nshapes)]
                            all_shape_ids.remove(jj+1)  # list of all elements except jj
                            new_id = numpy.random.choice(all_shape_ids, 1)  # choose an index except jj
                            new_id = new_id[0]
                        else:
                            new_id = rand_id # a very large number to indicate draw a random shape (below)
                    else:
                        new_id = shape_id

                    # during num_cells experiment, non-diagnostic stim involves random number of cells
                    if experiment == 'dnum_cells' or experiment == 'dnum_cells_invalid':
                        if set_type == 'test_nodiag':
                            rand_num_cells = numpy.random.randint(nshapes)
                            shape_ncells = set_ncells[rand_num_cells]

                    if experiment == 'dcol_seg_noshape' or set_type == 'test_inv':
                        segments = gen_rand_shape(seg_len_cue, shape_ncells[0], shape_ncells[1])
                    else:
                        if new_id == 1:
                            segments = gen_shape1(seg_len_cue, shape_ncells[0], shape_ncells[1])
                        elif new_id == 2:
                            segments = gen_shape2(seg_len_cue, shape_ncells[0], shape_ncells[1])
                        elif new_id == 3:
                            segments = gen_shape3(seg_len_cue, shape_ncells[0], shape_ncells[1])
                        elif new_id == 4:
                            segments = gen_shape4(seg_len_cue, shape_ncells[0], shape_ncells[1])
                        elif new_id == 5:
                            segments = gen_shape5(seg_len_cue, shape_ncells[0], shape_ncells[1])
                        elif new_id == rand_id:
                            segments = gen_rand_shape(seg_len_cue, shape_ncells[0], shape_ncells[1])

                    # Unless patch is diagnostic, i.e. if testing for only unique color,
                    # sample different loc for each training item
                    if experiment != 'dpatch' and experiment != 'dpatch_invalid' and \
                       experiment != 'dcol_oneloc_invalid':
                        ### DEBUG
                        # minmax_locs = (280, 320) # Allows larger cell size without clipping
                        ### DEBUG
                        shape_loc = (numpy.random.randint(minmax_locs[0], minmax_locs[1]),
                                     numpy.random.randint(minmax_locs[0], minmax_locs[1]))
                    # if set_type == 'test_nodiag' and experiment == 'dunicol': # each item has random (unif) color
                    #     shape_color = (numpy.random.randint(max_col),
                    #                    numpy.random.randint(max_col),
                    #                    numpy.random.randint(max_col))
                    #     shape_color = tuple([cc * int(256 / max_col) for cc in shape_color]) # rescale
                    if set_type == 'test_nodiag': # use random color in nodiag condition
                        if use_palette is True:
                            shape_color = random.choice(avail_cols)
                        else:
                            shape_color = (numpy.random.randint(max_col),
                                           numpy.random.randint(max_col),
                                           numpy.random.randint(max_col))
                            shape_color = tuple([cc * int(256 / max_col) for cc in shape_color])  # rescale
                    if experiment == 'dcell_size' or experiment == 'dcell_size_invalid' or \
                       experiment == 'dnum_cells' or experiment == 'dnum_cells_invalid':
                        if use_palette is True:
                            shape_color = random.choice(avail_cols)
                        else:
                            shape_color = (numpy.random.randint(max_col),
                                           numpy.random.randint(max_col),
                                           numpy.random.randint(max_col))
                            shape_color = tuple([cc * int(256 / max_col) for cc in shape_color])  # rescale

                    if upright == True:
                        orient = 0  # for rotation net
                    else:
                        orient = numpy.random.choice(orientations)  # Choose a random orientation
                    flip_lr = False
                    flip_tb = False
                    # if numpy.random.rand() > 0.5:
                    #     flip_lr = True
                    # if numpy.random.rand() > 0.5:
                    #     flip_tb = True

                    if experiment == 'dcell_size' or experiment == 'dcell_size_invalid':
                        cell_size = shape_size

                    (im, im_patches) = draw_shape(segments, shape_color, shape_loc, shift_type, orient, flip_lr,
                                                  flip_tb, cell_size, imsize, cat_colors, unif_col, unif_seg,
                                                  max_col, all_same, im_patches, col_del, shape_freq, avail_cols,
                                                  jit, shift_centre)
                    if (exclude_locs is True): # ensure no part of shape overlaps cat_locs[jj]
                        tl = (cat_locs[jj][0], cat_locs[jj][1]) # top-left
                        tr = (cat_locs[jj][0]+cell_size+1, cat_locs[jj][1]) # top-right
                        bl = (cat_locs[jj][0], cat_locs[jj][1]+cell_size+1) # bottom-left
                        br = (cat_locs[jj][0]+cell_size+1, cat_locs[jj][1]+cell_size+1) # bottom-right
                        num_tries = 0  # so that we don't get stuck in this loop
                        max_tries = 10
                        while ((im.getpixel(tl) != (255, 255, 255) or
                                im.getpixel(tr) != (255, 255, 255) or
                                im.getpixel(bl) != (255, 255, 255) or
                                im.getpixel(br) != (255, 255, 255)) and
                               num_tries < max_tries):
                            # Sample shape_loc again to prevent infinte loop if shape_loc overlaps cat_locs[jj]
                            if experiment != 'dpatch' and experiment != 'dpatch_invalid' and \
                               experiment != 'dcol_oneloc_invalid':
                                shape_loc = (numpy.random.randint(minmax_locs[0], minmax_locs[1]),
                                             numpy.random.randint(minmax_locs[0], minmax_locs[1]))
                            # Draw a new shape till no overlap with cat_locs[jj]
                            (im, im_patches) = draw_shape(segments, shape_color, shape_loc, shift_type, orient, flip_lr,
                                                          flip_tb, cell_size, imsize, cat_colors, unif_col, unif_seg,
                                                          max_col, all_same, im_patches, col_del, shape_freq, avail_cols,
                                                          jit, shift_centre)
                            num_tries += 1

                    ### Save image
                    im_file = os.path.join(im_folder, 'shape_' + str(shape_id), str(ii) + '.png')
                    if not os.path.exists(os.path.dirname(im_file)):
                        try:
                            os.makedirs(os.path.dirname(im_file))
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                    im.save(im_file)

                # DEBUG START - check overlap in patches of diff categories
                # im_patches_file = os.path.join(im_folder, 'patches_shape_' + str(shape_id) + '.png')
                # im_patches.save(im_patches_file)
                # DEBUG END - check overlap in patches of diff categories
