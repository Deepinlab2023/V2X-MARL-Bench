"""Editable, deterministic V2X-MARL-Bench system figure.

Run with Python + reportlab. All coordinates are illustrative, not sampled data.
The desired/interfering edges are generated from the same channel assignments.
"""
from pathlib import Path
from copy import deepcopy
import math
from xml.etree import ElementTree as ET
from reportlab.graphics.shapes import Drawing, Group, Rect, Line, Path as RPath, Polygon, Circle, String
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = Path(__file__).resolve().parent
W, H = 1800, 1170
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
pdfmetrics.registerFont(TTFont('Arial', str(FONT_DIR / 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(FONT_DIR / 'Arial Bold.ttf')))
INK, MUTED, RULE = '#172D46', '#607084', '#D7DFE7'
ROAD, MEDIAN, WHITE = '#EEF2F5', '#E1E7EB', '#FFFFFF'
COL = {1: '#2268AD', 2: '#12867E', 3: '#7954A5', 4: '#AD7B26'}
PALE = {1: '#EDF4FB', 2: '#ECF7F5', 3: '#F3EFF8', 4: '#FAF5E9'}
INTERF = '#C45143'

# An explicit allocation, used consistently in every panel.
CHANNEL = {'T1': 1, 'T2': 1, 'T3': 2, 'T4': 3,
           'U1': 1, 'U2': 2, 'U3': 3, 'U4': 4}
PAIRS = {'T1': 'R1', 'T2': 'R2', 'T3': 'R3', 'T4': 'R4'}
COCHANNEL = [('T2', 'R1'), ('U1', 'R1'), ('T1', 'R2'),
             ('U1', 'R2'), ('T1', 'BS'), ('T2', 'BS')]
assert len(COCHANNEL) == 6
assert all(CHANNEL[src] == 1 for src, _ in COCHANNEL)

d = Drawing(W, H)
d.add(Rect(0, 0, W, H, fillColor=HexColor(WHITE), strokeColor=None))

def c(value):
    return HexColor(value) if value else None

def rect(x, y, w, h, fill, stroke=None, sw=1, r=0):
    d.add(Rect(x, H-y-h, w, h, rx=r, ry=r, fillColor=c(fill), strokeColor=c(stroke), strokeWidth=sw))

def circle(x, y, radius, fill, stroke=None, sw=1):
    d.add(Circle(x, H-y, radius, fillColor=c(fill), strokeColor=c(stroke), strokeWidth=sw))

def text(x, y, s, size=22, color=INK, bold=False, anchor='start'):
    d.add(String(x, H-y, s, fontName='Arial-Bold' if bold else 'Arial', fontSize=size,
                 fillColor=c(color), textAnchor=anchor))

def line(x1, y1, x2, y2, color=RULE, sw=1, dash=None):
    d.add(Line(x1, H-y1, x2, H-y2, strokeColor=c(color), strokeWidth=sw,
               strokeDashArray=dash, strokeLineCap=1))

def path(points, color, sw=2, dash=None):
    p = RPath(fillColor=None, strokeColor=c(color), strokeWidth=sw,
              strokeDashArray=dash, strokeLineCap=1, strokeLineJoin=1)
    for command, *args in points:
        if command == 'M': p.moveTo(args[0], H-args[1])
        elif command == 'L': p.lineTo(args[0], H-args[1])
        elif command == 'C': p.curveTo(args[0], H-args[1], args[2], H-args[3], args[4], H-args[5])
    d.add(p)

def arrow_head(x, y, dx, dy, color, size=11):
    a = math.atan2(dy, dx)
    pts = [(x, y), (x-size*math.cos(a)+size*.45*math.sin(a), y-size*math.sin(a)-size*.45*math.cos(a)),
           (x-size*math.cos(a)-size*.45*math.sin(a), y-size*math.sin(a)+size*.45*math.cos(a))]
    d.add(Polygon([v for px,py in pts for v in (px,H-py)], fillColor=c(color), strokeColor=None))

def arrow(x1, y1, x2, y2, color=INK, sw=2.6, dash=None, size=10):
    line(x1, y1, x2, y2, color, sw, dash)
    arrow_head(x2, y2, x2-x1, y2-y1, color, size)

def curve(x1,y1,c1x,c1y,c2x,c2y,x2,y2,color,sw=3,dash=None,halo=False,size=11):
    pts = [('M',x1,y1), ('C',c1x,c1y,c2x,c2y,x2,y2)]
    if halo: path(pts, WHITE, sw+5)
    path(pts, color, sw, dash)
    arrow_head(x2, y2, x2-c2x, y2-c2y, color, size)

def label(x,y,s,color=INK,size=19,fill=WHITE):
    width = pdfmetrics.stringWidth(s,'Arial-Bold',size)+14
    rect(x-width/2,y-18,width,25,fill,r=4)
    text(x,y,s,size,color,True,'middle')

def car(x,y,name,channel,role='Tx',direction=1):
    color = COL[channel]
    # Roof silhouette with wheels and distinct front windscreen.
    for wx in (-22,22):
        for wy in (-17,13): rect(x+wx-6,y+wy,12,4,INK,r=1.4)
    rect(x-37,y-14,74,28,color if role != 'Rx' else WHITE,color,2,r=9)
    rect(x-16,y-10,33,20,PALE[channel] if role=='Rx' else color,WHITE if role!='Rx' else color,.8,r=5)
    front = x+13 if direction==1 else x-21
    rect(front,y-9,8,18,WHITE if role!='Rx' else PALE[channel],r=2)
    rear = x-26 if direction==1 else x+21
    rect(rear,y-8,5,16,PALE[channel],r=1)
    if role=='V2I':
        circle(x+1,y,3,WHITE)
    label(x,y-24,name,color,19)

def tower(x,y,scale=1):
    # y marks the receive point; truss antenna stands below it.
    def l(a,b,cc,dd,sw=2.6): line(x+a*scale,y+b*scale,x+cc*scale,y+dd*scale,INK,sw*scale)
    l(0,0,-24,77); l(0,0,24,77); l(-28,78,28,78)
    l(-8,28,10,28,2); l(-15,51,16,51,2)
    l(-8,28,16,51,1.8); l(8,28,-16,51,1.8)
    l(-15,51,24,77,1.8); l(15,51,-24,77,1.8)
    l(0,-10,0,8,3); circle(x,y-11*scale,4*scale,INK)
    for sign in (-1,1):
        for r in (15,25):
            path([('M',x+sign*r*.67*scale,y-r*.80*scale),
                  ('C',x+sign*r*1.2*scale,y-r*.55*scale,x+sign*r*1.2*scale,y+r*.1*scale,x+sign*r*.67*scale,y+r*.38*scale)],INK,1.6*scale)

def heading(x,y,letter,title):
    text(x,y,f'({letter})',25,MUTED,True)
    text(x+48,y,title,27,INK,True)

def node(x,y,s,filled=True,color=COL[1],w=64):
    rect(x-w/2,y-18,w,36,color if filled else WHITE,color,2,r=7)
    text(x,y+7,s,21,WHITE if filled else color,True,'middle')

# Header: deliberately minimal so the illustration also works on a project page.
text(50,48,'V2X-MARL-BENCH',18,MUTED,True)
text(50,103,'C-V2X sidelink resource allocation',43,INK,True)
text(1750,61,'SYSTEM MODEL',17,MUTED,True,'end')
text(1750,95,'Shared uplink spectrum',21,MUTED,False,'end')
line(50,128,1750,128,RULE,1.2)

# A. Deployment and desired communication links.
a_start = len(d.contents)
heading(50,177,'a','Highway deployment')
text(50,207,'Illustrative 4-agent allocation · three lanes in each direction',20,MUTED)
rect(50,232,1140,138,ROAD)
rect(50,404,1140,138,ROAD)
rect(50,370,1140,34,MEDIAN)
for yy in (232,370,404,542): line(50,yy,1190,yy,'#A6B4C1',1.5)
for yy in (278,324,450,496): line(50,yy,1190,yy,'#BAC6D0',1.2,[12,12])
arrow(1170,380,1138,380,MUTED,1.6,size=7)
arrow(1138,394,1170,394,MUTED,1.6,size=7)

# Uplink arrows first, with white clearance around each stroke.
curve(160,271,132,533,255,619,592,647,COL[3],2.5,halo=True)
curve(340,363,334,528,466,585,608,639,COL[4],2.5,halo=True)
curve(120,535,243,613,392,653,591,659,COL[1],2.8,halo=True)
curve(1110,443,1138,591,863,643,650,651,COL[2],2.8,halo=True)

# Four intended V2V links. T1 and T2 reuse the same subchannel.
curve(703,255,646,254,626,300,550,301,COL[2],3.4,halo=True)
curve(1033,347,973,348,969,301,900,301,COL[3],3.4,halo=True)
curve(247,427,318,427,332,473,410,473,COL[1],3.4,halo=True)
curve(797,519,874,519,904,473,990,473,COL[1],3.4,halo=True)

for args in [(740,255,'T3',2,'Tx',-1),(510,301,'R3',2,'Rx',-1),
             (1070,347,'T4',3,'Tx',-1),(860,301,'R4',3,'Rx',-1),
             (160,255,'U3',3,'V2I',-1),(340,347,'U4',4,'V2I',-1),
             (210,427,'T1',1,'Tx',1),(450,473,'R1',1,'Rx',1),
             (760,519,'T2',1,'Tx',1),(1030,473,'R2',1,'Rx',1),
             (120,519,'U1',1,'V2I',1),(1110,427,'U2',2,'V2I',1)]:
    car(*args)

tower(620,657,.77)
text(665,697,'Base station',22,INK,True)
text(665,723,'Receives the four V2I uplinks',18,MUTED)
text(50,675,'Desired links shown here.',18,MUTED)
text(50,700,'Interference is expanded in (c).',18,MUTED)
text(50,725,'Schematic; not to scale.',17,MUTED)
text(50,759,'T: V2V transmitter / agent     R: paired V2V receiver     U: V2I user',20,MUTED)
a_end = len(d.contents)

# B. Channel occupancy matches the diagram and the interference expansion.
line(1230,157,1230,765,RULE,1)
heading(1270,177,'b','Spectrum reuse')
text(1270,207,'Four orthogonal subchannels, 1 MHz each',20,MUTED)
reuse = {1:'T1 → R1     T2 → R2',2:'T3 → R3',3:'T4 → R4',4:'No V2V reuse in this example'}
for m in range(1,5):
    yy = 238 + (m-1)*107
    rect(1270,yy,480,92,PALE[m],r=7)
    rect(1270,yy,5,92,COL[m],r=2)
    text(1294,yy+32,f'SC {m}',22,COL[m],True)
    text(1726,yy+32,f'U{m} → BS',22,COL[m],True,'end')
    text(1294,yy+68,reuse[m],21 if m<4 else 19,INK if m<4 else MUTED)
text(1270,705,'One fixed V2I uplink per subchannel.',20,INK)
text(1270,736,'V2V agents select which one to reuse.',20,MUTED)

# C. Full interference relationships for the channel reused by two V2V agents.
line(50,792,1750,792,RULE,1.2)
c_start = len(d.contents)
heading(50,841,'c','Co-channel interference')
text(1190,841,'SC 1 expanded',20,COL[1],True,'end')

targets = [('R1','T1',('T2','U1')),('R2','T2',('T1','U1')),('BS','U1',('T1','T2'))]
for idx,(rx,desired,interferers) in enumerate(targets):
    x=50+idx*390
    if idx>0: line(x-16,880,x-16,1045,RULE,.8)
    text(x+165,886,'At the base station' if rx=='BS' else f'At V2V receiver {rx}',20,INK,True,'middle')
    sx, tx = x+53,x+290
    sy, my, by = 930,981,1032
    curve(sx+33,sy,sx+107,sy,tx-90,my-11,tx-36,my-8,INTERF,2.5,[7,6],size=9)
    arrow(sx+33,my,tx-36,my,COL[1],3,size=10)
    curve(sx+33,by,sx+107,by,tx-90,my+11,tx-36,my+8,INTERF,2.5,[7,6],size=9)
    node(sx,sy,interferers[0]); node(sx,my,desired); node(sx,by,interferers[1])
    node(tx,my,rx,False,w=72)

arrow(70,1093,124,1093,COL[1],3,size=9)
text(138,1100,'Desired signal',19,INK)
arrow(325,1093,379,1093,INTERF,2.5,[7,6],size=9)
text(394,1100,'Interference',19,INK)
text(1190,1100,'All six interfering paths on SC 1 are shown.',18,MUTED,False,'end')
c_end = len(d.contents)

# D. Action space: the selected action changes both occupancy and interference.
line(1230,821,1230,1110,RULE,1)
heading(1270,841,'d','Agent decision')
text(1270,890,'Each V2V transmitter selects',22,INK)
text(1270,931,'Subchannel',19,MUTED)
text(1485,931,'SC 1, 2, 3 or 4',22,INK,True)
text(1270,976,'Transmit power',19,MUTED)
text(1485,976,'23, 10 or 5 dBm',22,INK,True)
line(1270,998,1750,998,RULE,1)
text(1270,1038,'or chooses no transmission',22,INK)
text(1270,1097,'4 × 3 + 1 = 13 actions',29,COL[1],True)

line(50,1130,1750,1130,RULE,1)
text(50,1157,'Benchmark: 4, 8 or 16 V2V agents · 4 V2I users · 1 ms decision steps in SIG / POSIG',17,MUTED)
text(1750,1157,'Link colors identify subchannels.',17,MUTED,False,'end')

OUT.mkdir(parents=True,exist_ok=True)
renderPDF.drawToFile(d,str(OUT/'v2x_system_model.pdf'),title='V2X-MARL-Bench: C-V2X sidelink resource allocation',author='V2X-MARL-Bench')
renderSVG.drawToFile(d,str(OUT/'v2x_system_model.svg'))

# A compact, 180 mm wide, two-panel version for double-column paper layout.
# Reuse the very same vector objects, so no network relationships can drift.
paper = Drawing(1240,1070)
paper.add(Rect(0,0,1240,1070,fillColor=c(WHITE),strokeColor=None))
g = Group()
g.translate(0,20)  # original y becomes y-120 in the compact 1070-high drawing
for i in list(range(a_start,a_end)) + list(range(c_start,c_end)):
    item=deepcopy(d.contents[i])
    if isinstance(item,String):
        if item.text == '(c)': item.text='(b)'
        if item.text == 'Interference is expanded in (c).': item.text='Interference is expanded in (b).'
    g.add(item)
paper.add(g)
paper.add(Line(50,1070-672,1190,1070-672,strokeColor=c(RULE),strokeWidth=1.2))
paper.add(Line(50,1070-1005,1190,1070-1005,strokeColor=c(RULE),strokeWidth=1))
for m in range(1,5):
    xx=50+(m-1)*144
    paper.add(Line(xx,1070-1035,xx+32,1070-1035,strokeColor=c(COL[m]),strokeWidth=3))
    paper.add(String(xx+44,1070-1042,f'SC {m}',fontName='Arial-Bold',fontSize=20,fillColor=c(COL[m])))
paper.add(String(1190,1070-1042,'4 subchannels × 3 powers + no transmission',fontName='Arial',fontSize=20,fillColor=c(MUTED),textAnchor='end'))
renderSVG.drawToFile(paper,str(OUT/'v2x_system_model_paper.svg'))
paper_pdf=deepcopy(paper)
paper_pdf.scale(510.24/1240,510.24/1240)
paper_pdf.width=510.24
paper_pdf.height=1070*510.24/1240
renderPDF.drawToFile(paper_pdf,str(OUT/'v2x_system_model_paper.pdf'),title='C-V2X system model',author='V2X-MARL-Bench')

# Add accessible metadata while leaving all drawing elements editable.
svg=OUT/'v2x_system_model.svg'
s=svg.read_text()
start=s.index('>',s.index('<svg'))+1
desc=('Four V2V pairs reuse four orthogonal V2I uplink subchannels. T1 and T2 share SC 1 with U1; '
      'T3 shares SC 2 with U2; T4 shares SC 3 with U3. U4 occupies SC 4. '
      'Six directed interfering paths on SC 1 are expanded by receiver: T2 and U1 to R1, '
      'T1 and U1 to R2, and T1 and T2 to the base station. '
      'Every V2V transmitter chooses one of four subchannels and three power levels, or no transmission.')
s=s[:start]+'\n<title>C-V2X sidelink resource allocation</title>\n<desc>'+desc+'</desc>'+s[start:]
svg.write_text(s)
ET.parse(svg)
print(f'Created {OUT}/v2x_system_model.[svg,pdf]')
