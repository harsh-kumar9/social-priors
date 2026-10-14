"""Soft world palette: a modest contrast increase over the original pastels.

Preserves blue/yellow/gray/red assignments and small black seed dots. Original-
model bars are white with dark outlines, distinct from task-instrumental gray.
Direct labels, ordering, outlines and existing evaluation hatches
carry information independently of color. This custom palette is not claimed
as a universally distinguishable palette for every form of color vision.
"""
from matplotlib.patches import Patch

FILL=dict(P='#C7D9E9',S='#F0DFAE',N='#D6D6D2',A='#E7C2BB',
          base='#FFFFFF',MIX='#DDDDDD',REPLAY='#F4F4F4',NONE='#FFFFFF',other='#FFFFFF')
LINE=dict(P='#37668F',S='#876A16',N='#5F5F5B',A='#A14C42')
MARKER=dict(P='o',S='o',N='o',A='o')
INK='#242424'
HANDLER_MAP={}

def marker_for_fill(fill):
    return 'o'

def world_handle(w):
    return Patch(facecolor=FILL[w],edgecolor=INK,lw=.6)

def segment_text_color(w):
    return INK
