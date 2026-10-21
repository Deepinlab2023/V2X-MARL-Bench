"""Reproducible, editable topology triptych from V2X-MARL-Bench rep9.

Requires reportlab; PNG preview is rendered from the PDF using pdftoppm.
Centers use the original CSV coordinates. X scales differ between panels;
lane geometry uses a common schematic vertical scale. Vehicle icons are symbols.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String, Group, Path as RPath
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = Path(__file__).resolve().parent
REPO = Path('/Users/swang66/Documents/GitHub/V2X-MARL-Bench')
SOURCE = REPO / 'Environment/SUMOData/eval_sets/rep9_k4.csv'
META = REPO / 'Environment/SUMOData/eval_sets/eval_sets_meta.csv'
W, H = 1440, 420
BLUE, GOLD = '#28668E', '#B7802A'
INK, MUTED, ROAD, LINE = '#233747', '#62717B', '#EFF2F4', '#B7C3CB'
FONT = Path('/System/Library/Fonts/Supplemental')
pdfmetrics.registerFont(TTFont('Arial', str(FONT / 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(FONT / 'Arial Bold.ttf')))
d = Drawing(W, H)

def color(s): return HexColor(s) if s else None

def rect(x,y,w,h,fill,stroke=None,sw=1,r=0):
    d.add(Rect(x,H-y-h,w,h,rx=r,ry=r,fillColor=color(fill),strokeColor=color(stroke),strokeWidth=sw))

def line(x1,y1,x2,y2,c=LINE,sw=1,dash=None):
    d.add(Line(x1,H-y1,x2,H-y2,strokeColor=color(c),strokeWidth=sw,strokeDashArray=dash,strokeLineCap=1))

def text(x,y,s,size=22,c=INK,bold=False,anchor='start'):
    d.add(String(x,H-y,s,fontName='Arial-Bold' if bold else 'Arial',fontSize=size,fillColor=color(c),textAnchor=anchor))

def car(x,y,role):
    # Deliberately constant symbol size, independent of each horizontal zoom.
    c = GOLD if role == 'V2I' else BLUE
    fill = '#FFFFFF' if role == 'Rx' else c
    for wx in (-8,8):
        for wy in (-7,6): rect(x+wx-3,y+wy,6,2,INK,r=.7)
    rect(x-15,y-6,30,12,fill,c,1.6,r=3.5)
    rect(x-6,y-4,12,8,'#FFFFFF' if role!='Rx' else '#EAF2F7',c,.7,r=1.5)

def arrow(x1,y1,x2,y2,bend=0):
    dx,dy = x2-x1,y2-y1
    length = math.hypot(dx,dy)
    ux,uy = dx/length,dy/length
    # Intersect icon's padded bounding box instead of letting arrows cover cars.
    trim = min(18/abs(ux) if ux else 1e6, 10/abs(uy) if uy else 1e6)
    assert 2*trim < length, (x1,y1,x2,y2)
    ax,ay,bx,by = x1+ux*trim,y1+uy*trim,x2-ux*trim,y2-uy*trim
    if bend:
        # A display-only bend keeps links clear of unrelated vehicle symbols.
        c1x,c1y=x1+dx/3-uy*bend,y1+dy/3+ux*bend
        c2x,c2y=x1+dx*2/3-uy*bend,y1+dy*2/3+ux*bend
        def edge(nx,ny,cx,cy):
            vx,vy=cx-nx,cy-ny
            ll=math.hypot(vx,vy); vx,vy=vx/ll,vy/ll
            tt=min(18/abs(vx) if vx else 1e6,10/abs(vy) if vy else 1e6)
            return nx+vx*tt,ny+vy*tt
        ax,ay=edge(x1,y1,c1x,c1y)
        bx,by=edge(x2,y2,c2x,c2y)
        for cc,sw in [('#FFFFFF',5),('#657F91',1.8)]:
            p=RPath(fillColor=None,strokeColor=color(cc),strokeWidth=sw,strokeLineCap=1)
            p.moveTo(ax,H-ay); p.curveTo(c1x,H-c1y,c2x,H-c2y,bx,H-by); d.add(p)
        ll=math.hypot(bx-c2x,by-c2y); ux,uy=(bx-c2x)/ll,(by-c2y)/ll
    else:
        line(ax,ay,bx,by,'#FFFFFF',5)
        line(ax,ay,bx,by,'#657F91',1.8)
    size=6.5
    pts=[(bx,by),(bx-size*ux+size*.45*uy,by-size*uy-size*.45*ux),
         (bx-size*ux-size*.45*uy,by-size*uy+size*.45*ux)]
    d.add(Polygon([v for x,y in pts for v in (x,H-y)],fillColor=color('#657F91'),strokeColor=None))

rows = list(csv.DictReader(SOURCE.open()))
meta = list(csv.DictReader(META.open()))
settings = [
    dict(sid=1237,scenario='etsi1',letter='a',name='Low density',label='ETSI #1',density=35,xlim=(-200,200),bar=100),
    dict(sid=19092,scenario='etsi3',letter='b',name='Medium density',label='ETSI #3',density=123,xlim=(-65,65),bar=20),
    dict(sid=29132,scenario='etsi6',letter='c',name='High density',label='ETSI #6',density=500,xlim=(-16,16),bar=5),
]
rect(0,0,W,H,'#FFFFFF')
provenance=[]
selected=[]
for idx,s in enumerate(settings):
    nodes=[r for r in rows if r['snapshot_id']==str(s['sid'])]
    assert len(nodes)==12 and {r['scenario'] for r in nodes}=={s['scenario']}
    assert all(sum(r['role']==role for r in nodes)==4 for role in ('Tx','Rx','V2I'))
    m=next(r for r in meta if r['k']=='4' and r['set']=='rep9' and r['snapshot_id']==str(s['sid']))
    assert m['band']=='near'
    left=28+idx*480
    width=424
    low,high=s['xlim']
    def pos(r): return left+(float(r['x'])-low)/(high-low)*width, 272-6*float(r['y'])
    text(left,30,f"({s['letter']}) {s['name']}",27,bold=True)
    text(left,61,f"{s['label']}  /  {s['density']} veh/km",22,c=MUTED)
    for top in (92,200):
        rect(left,top,width,72,ROAD)
        line(left,top,left+width,top,'#CCD5DB',1.1)
        line(left,top+72,left+width,top+72,'#CCD5DB',1.1)
        for lane in (24,48): line(left,top+lane,left+width,top+lane,'#BDC8D0',1,dash=[7,7])
    links=[]
    for pair in range(4):
        tx=next(r for r in nodes if r['role']=='Tx' and r['pair_id']==str(pair))
        rx=next(r for r in nodes if r['role']=='Rx' and r['pair_id']==str(pair))
        bend={(19092,0):35,(29132,2):22}.get((s['sid'],pair),0)
        arrow(*pos(tx),*pos(rx),bend=bend)
        links.append(dict(pair_id=pair,tx_node=int(tx['node_id']),rx_node=int(rx['node_id']),
                          distance_m=math.hypot(float(tx['x'])-float(rx['x']),float(tx['y'])-float(rx['y'])),display_bend=bool(bend)))
    for n in nodes:
        x,y=pos(n)
        assert left+16<x<left+width-16, (s,n)
        car(x,y,n['role'])
    bar_px=width*s['bar']/(high-low)
    bx=left+width-bar_px
    by=307
    line(bx,by,bx+bar_px,by,INK,2)
    line(bx,by-4,bx,by+4,INK,1.4)
    line(bx+bar_px,by-4,bx+bar_px,by+4,INK,1.4)
    text(bx+bar_px/2,334,f"{s['bar']} m",21,anchor='middle')
    provenance.append(dict(**s,band=m['band'],nodes=12,v2v_pairs=4,v2i_users=4,links=links))
    selected.extend(nodes)

# One shared legend. Color plus open/filled vehicle bodies distinguish roles.
car(240,367,'Tx'); text(265,374,'V2V transmitter',22)
car(540,367,'Rx'); text(565,374,'V2V receiver',22)
car(810,367,'V2I'); text(835,374,'V2I user',22)
arrow(1010,367,1080,367); text(1100,374,'V2V link',22)
text(W/2,409,'Selected communication nodes. Horizontal scales differ; vehicle symbols and lane spacing are schematic.',20,c=MUTED,anchor='middle')

svg=OUT/'v2x_density_topologies.svg'
renderSVG.drawToFile(d,str(svg))
# Size the PDF for a 180 mm, two-column research figure.
scale=(180/25.4*72)/W
pdf=Drawing(W*scale,H*scale)
g=Group(); g.add(d); g.scale(scale,scale); pdf.add(g)
renderPDF.drawToFile(pdf,str(OUT/'v2x_density_topologies.pdf'))
subprocess.run(['/opt/homebrew/bin/pdftoppm','-png','-singlefile','-scale-to','2880',
                str(OUT/'v2x_density_topologies.pdf'),str(OUT/'v2x_density_topologies')],check=True)
with (OUT/'selected_nodes.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(selected)
(OUT/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),panels=provenance,
    transform='Original x and y are used for symbol centers; each panel has an independent linear x scale and common schematic y scale. No nodes or links omitted.',
    omissions='BS and all V2I uplink / interference edges are omitted to show selected V2V topology only.'),indent=2))
(OUT/'caption.txt').write_text('Representative communication topologies sampled from three SUMO traffic-density scenarios: ETSI #1 (35 veh/km), ETSI #3 (123 veh/km), and ETSI #6 (500 veh/km). Each panel shows four V2V transmitter-receiver pairs and four V2I users from the near-BS subset of rep9_k4. Arrows indicate V2V links. Horizontal scales differ; vehicle symbols and lane spacing are schematic. Only selected communication vehicles are shown, rather than all traffic. Base stations, V2I uplink edges, and interference edges are omitted for clarity.\n')
print(json.dumps({'outputs':[str(p) for p in OUT.iterdir() if p.is_file()], 'nodes':len(selected),'v2v_links':sum(len(p['links']) for p in provenance)},indent=2))
