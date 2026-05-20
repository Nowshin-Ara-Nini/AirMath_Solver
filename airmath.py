"""
AirMath v4 — Fixed Digit Recognition + Smooth Air Drawing
==========================================================
DRAW:   ☝  Only index UP, others curled  → draws
LIFT:   ✋  2+ fingers up                 → pen up
EVAL:   🤏  Pinch OR press SPACE          → evaluate

C=clear  Z=undo  S=skeleton  1-5=color  +/-=size  Q=quit
"""

import cv2, mediapipe as mp, numpy as np
import sympy, math, time, re, base64, zlib, os
import math_ocr
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application)
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── CONFIG ────────────────────────────────────────────────────
WIN       = "AirMath v4"
CAM_IDX   = 0
CAM_W, CAM_H = 1280, 720
PANEL_W   = 380
HEAD_H    = 58
FOOT_H    = 72
BRUSH_DEF = 14
PINCH_THR = 0.075
DIGITS    = "0123456789"
COLORS    = [(0,255,150),(255,255,255),(100,180,255),(255,80,200),(50,220,255)]
IDX_TIP,IDX_MCP = 8,5
MID_TIP,MID_MCP = 12,9
RNG_TIP,RNG_MCP = 16,13
PNK_TIP,PNK_MCP = 20,17
THM_TIP         = 4

# ── EMBEDDED CENTROID CLASSIFIER ─────────────────────────────
_CDATA  = "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"
_CSHAPE = (10,136)

def _load_centroids():
    raw = zlib.decompress(base64.b64decode(_CDATA))
    return np.frombuffer(raw, dtype=np.float32).reshape(_CSHAPE)

_CENTROIDS = _load_centroids()
_CNN_MODEL = None
_CNN_LABELS = "0123456789+-*/"
try:
    import tensorflow as tf
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "air_math_model.keras")
    if os.path.exists(_MODEL_PATH):
        _CNN_MODEL = tf.keras.models.load_model(_MODEL_PATH)
except Exception:
    _CNN_MODEL = None

def _preprocess(img, size=32):
    if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thr=cv2.threshold(img,30,255,cv2.THRESH_BINARY)
    coords=cv2.findNonZero(thr)
    if coords is None: return np.zeros((size,size),np.float32)
    x,y,w,h=cv2.boundingRect(coords)
    if w==0 or h==0: return np.zeros((size,size),np.float32)
    crop=thr[y:y+h,x:x+w]
    pad=max(w,h)//8+2
    crop=cv2.copyMakeBorder(crop,pad,pad,pad,pad,cv2.BORDER_CONSTANT)
    return cv2.resize(crop,(size,size)).astype(np.float32)/255.0

def _features(f):
    grid=cv2.resize((f*255).astype(np.uint8),(8,8)).astype(np.float32)/255.0
    rows=f.mean(axis=1); cols=f.mean(axis=0)
    q=[f[:16,:16].mean(),f[:16,16:].mean(),f[16:,:16].mean(),f[16:,16:].mean()]
    thr8=(f*255).astype(np.uint8)
    _,thr8=cv2.threshold(thr8,127,255,cv2.THRESH_BINARY)
    _,h=cv2.findContours(thr8,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes=min(sum(1 for x in h[0] if x[3]>=0),3) if h is not None and len(h)>0 else 0
    m=cv2.moments(thr8)
    cx=m["m10"]/m["m00"]/32 if m["m00"]>0 else 0.5
    cy=m["m01"]/m["m00"]/32 if m["m00"]>0 else 0.5
    return np.concatenate([grid.flatten(),rows,cols,q,[holes/3.,cx,cy,f.mean()]]).astype(np.float32)

def classify_digit(roi_gray) -> Tuple[str, float]:
    """Classify a single digit ROI. Prefer CNN if available, fallback to centroid."""
    pp = _preprocess(roi_gray)
    if _CNN_MODEL is not None:
        try:
            x = cv2.resize((pp * 255).astype(np.uint8), (28, 28))
            x = (x.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
            probs = _CNN_MODEL.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            lbl = _CNN_LABELS[idx]
            if lbl in DIGITS:
                return lbl, float(probs[idx])
        except Exception:
            pass
    feat = _features(pp)
    norm = np.linalg.norm(_CENTROIDS,axis=1)*np.linalg.norm(feat)+1e-9
    sims = _CENTROIDS @ feat / norm
    idx  = int(np.argmax(sims))
    conf = float((sims[idx]+1)/2)
    return DIGITS[idx], conf

def detect_operator(roi32, comp) -> Optional[str]:
    h,w   = comp["h"],comp["w"]
    aspect= w/max(h,1)
    img   = cv2.resize(roi32,(32,32)).astype(float)
    # Normalize to [0,1] for robust thresholds.
    img01 = img / 255.0
    hr = img01[12:20,:].mean()
    vr = img01[:,12:20].mean()
    cor=(img01[:10,:10].mean()+img01[:10,22:].mean()+img01[22:,:10].mean()+img01[22:,22:].mean())/4
    if aspect>2.8 and h<50 and hr>0.18 and vr<0.10: return "-"
    if 0.6<aspect<1.6 and hr>0.20 and vr>0.20 and cor<0.20: return "+"
    if aspect>2.0 and h<45 and hr>0.12 and vr<0.12: return "/"
    rows=[i for i,v in enumerate(img01.mean(axis=1)) if v>0.22]
    if len(rows)>=2 and (max(rows)-min(rows))>8 and aspect>1.4: return "="
    return None

# ── OCR PIPELINE ─────────────────────────────────────────────
class OCR:
    def recognize(self, canvas_bgr) -> Tuple[str,float]:
        gray=cv2.cvtColor(canvas_bgr,cv2.COLOR_BGR2GRAY)
        _,thr=cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        dilated=cv2.dilate(thr,kernel,iterations=2)
        n,_,stats,cents=cv2.connectedComponentsWithStats(dilated,connectivity=8)
        if n<=1: return "",0.0
        blobs=[]
        for i in range(1,n):
            a=stats[i,cv2.CC_STAT_AREA]
            if a<200: continue
            blobs.append({"x":stats[i,cv2.CC_STAT_LEFT],"y":stats[i,cv2.CC_STAT_TOP],
                          "w":stats[i,cv2.CC_STAT_WIDTH],"h":stats[i,cv2.CC_STAT_HEIGHT],"area":a})
        if not blobs: return "",0.0
        blobs.sort(key=lambda b:b["x"])
        chars=self._group(blobs)
        tokens,confs=[],[]
        for ch in chars:
            roi=self._crop(thr,ch)
            op=detect_operator(roi,ch)
            if op: tokens.append(op); confs.append(0.82)
            else:
                d,c=classify_digit(roi)
                tokens.append(d); confs.append(c)
        expr=re.sub(r"\s*([+\-*/=])\s*",r" \1 "," ".join(tokens))
        expr=re.sub(r"\s+"," ",expr).strip()
        return expr, float(np.mean(confs)) if confs else 0.0

    def _group(self,blobs):
        if not blobs: return blobs
        mh=float(np.median([b["h"] for b in blobs])); gap=max(30,int(mh*0.6))
        out=[dict(blobs[0])]
        for b in blobs[1:]:
            p=out[-1]
            if (b["x"]-(p["x"]+p["w"]))<gap and                (min(b["y"]+b["h"],p["y"]+p["h"])-max(b["y"],p["y"]))>mh*0.25:
                nx,ny=min(p["x"],b["x"]),min(p["y"],b["y"])
                nx2,ny2=max(p["x"]+p["w"],b["x"]+b["w"]),max(p["y"]+p["h"],b["y"]+b["h"])
                p.update(x=nx,y=ny,w=nx2-nx,h=ny2-ny,area=p["area"]+b["area"])
            else: out.append(dict(b))
        return out

    def _crop(self,thr,blob):
        pad=8
        x1=max(0,blob["x"]-pad); y1=max(0,blob["y"]-pad)
        x2=min(thr.shape[1],blob["x"]+blob["w"]+pad)
        y2=min(thr.shape[0],blob["y"]+blob["h"]+pad)
        r=thr[y1:y2,x1:x2]
        return cv2.resize(r,(32,32)) if r.size>0 else np.zeros((32,32),np.uint8)

# ── MATH EVALUATOR ────────────────────────────────────────────
class Evaluator:
    _R=[(r"[×xX]","*"),(r"÷","/"),(r"√(\d+)",r"sqrt(\1)"),
        (r"(\d)\^",r"\1**"),(r"\^","**"),(r"(\d)\(",r"\1*(")]
    def calc(self,text):
        expr=text.strip()
        if not expr: return "","Write something first",0.0
        for p,r in self._R: expr=re.sub(p,r,expr)
        if "=" in expr:
            expr = expr.split("=", 1)[0].strip()
        expr = re.sub(r"[+*/-]+$", "", expr).strip()
        expr = re.sub(r"([+*/-])\1+", r"\1", expr)
        try:
            tf=standard_transformations+(implicit_multiplication_application,)
            ld={k:getattr(sympy,k) for k in ["sqrt","pi","E","sin","cos","tan","log","Abs","exp"] if hasattr(sympy,k)}
            res=sympy.simplify(parse_expr(expr,transformations=tf,local_dict=ld))
            try:
                f=float(res); s=str(int(f)) if f==int(f) else f"{f:.8g}"
            except: s=str(res)
            return text.strip(),s,0.95
        except: pass
        try:
            sg={k:getattr(math,k) for k in dir(math) if not k.startswith("_")}
            sg["__builtins__"]={}
            v=eval(expr,sg)
            if isinstance(v,float) and v==int(v): v=int(v)
            return text.strip(),str(v),0.78
        except Exception as e: return text.strip(),f"Error: {e}",0.0

# ── GESTURE DETECTOR ─────────────────────────────────────────
class Gesture:
    def __init__(self): self._buf=deque(maxlen=4)
    # Relaxed "finger up" threshold to work across more cameras/lighting.
    def _up(self,lm,tip,mcp): return lm.landmark[tip].y < lm.landmark[mcp].y-0.02
    def update(self,lm):
        idx=self._up(lm,IDX_TIP,IDX_MCP)
        mid=self._up(lm,MID_TIP,MID_MCP)
        rng=self._up(lm,RNG_TIP,RNG_MCP)
        pnk=self._up(lm,PNK_TIP,PNK_MCP)
        n=sum([idx,mid,rng,pnk])
        th=lm.landmark[THM_TIP]; ix=lm.landmark[IDX_TIP]
        pinch=math.hypot(th.x-ix.x,th.y-ix.y)<PINCH_THR
        # Allow index up with at most one extra finger to reduce false "no draw".
        raw=idx and n<=2 and not pinch
        self._buf.append(raw)
        smooth=sum(self._buf)>=max(1,len(self._buf)//2)
        return smooth,pinch,n

# ── CANVAS (with interpolation) ──────────────────────────────
@dataclass
class Stroke:
    pts:  List[Tuple[int,int]]=field(default_factory=list)
    color:Tuple[int,int,int]=(0,255,150)
    size: int=14

class Canvas:
    def __init__(self,w,h):
        self.w,self.h=w,h; self.strokes:List[Stroke]=[]; self.cur:Optional[Stroke]=None
        self._surf=np.zeros((h,w,4),np.uint8); self._last=None
    def pen_down(self,c,s): self.cur=Stroke(color=c,size=s); self._last=None
    def draw(self,x,y):
        if not self.cur: return
        if self._last:
            dx,dy=x-self._last[0],y-self._last[1]; dist=math.hypot(dx,dy)
            if dist>8:
                for i in range(1,int(dist/6)):
                    t=i/(dist/6)
                    self.cur.pts.append((int(self._last[0]+dx*t),int(self._last[1]+dy*t)))
        self.cur.pts.append((x,y)); self._last=(x,y)
    def pen_up(self):
        if self.cur and len(self.cur.pts)>1: self.strokes.append(self.cur)
        self.cur=None; self._last=None
    def undo(self):
        if self.strokes: self.strokes.pop()
    def clear(self): self.strokes.clear(); self.cur=None; self._last=None
    def render(self):
        self._surf[:]=0
        for s in self.strokes: self._paint(s)
        if self.cur: self._paint(self.cur)
        return self._surf
    def _paint(self,s):
        pts=s.pts
        if not pts: return
        if len(pts)==1: cv2.circle(self._surf,pts[0],s.size//2,(*s.color,255),-1,cv2.LINE_AA); return
        for i in range(1,len(pts)): cv2.line(self._surf,pts[i-1],pts[i],(*s.color,255),s.size,cv2.LINE_AA)
        cv2.circle(self._surf,pts[0],s.size//2,(*s.color,255),-1,cv2.LINE_AA)
        cv2.circle(self._surf,pts[-1],s.size//2,(*s.color,255),-1,cv2.LINE_AA)
    def to_ocr(self):
        img=np.zeros((self.h,self.w,3),np.uint8)
        for s in self.strokes:
            for i in range(1,len(s.pts)): cv2.line(img,s.pts[i-1],s.pts[i],(255,255,255),s.size+6,cv2.LINE_AA)
        return img
    @property
    def has_content(self): return bool(self.strokes)

# ── UI ────────────────────────────────────────────────────────
class UI:
    F=cv2.FONT_HERSHEY_SIMPLEX; FP=cv2.FONT_HERSHEY_PLAIN
    BG=(10,10,18); PANEL=(15,15,26); BORDER=(38,38,58)
    GREEN=(0,255,150); PURPLE=(200,80,255); WHITE=(215,215,230)
    MUTED=(85,85,115); RED=(55,55,230); ORANGE=(30,130,255); YELLOW=(30,200,255)
    def __init__(self,fw,fh,pw,hh,fth):
        self.fw,self.fh=fw,fh; self.pw=pw; self.hh=hh; self.fth=fth
        self.ow=fw+pw; self.oh=fh+hh+fth
    def frame(self,cam,cbgra,st):
        out=np.full((self.oh,self.ow,3),self.BG,np.uint8)
        out[:self.hh,:]=self.PANEL
        cv2.line(out,(0,self.hh-1),(self.ow,self.hh-1),self.BORDER,1)
        cv2.putText(out,"Air",(18,40),self.F,1.15,self.GREEN,2,cv2.LINE_AA)
        cv2.putText(out,"Math",(85,40),self.F,1.15,self.PURPLE,2,cv2.LINE_AA)
        sc={"READY":self.GREEN,"DRAWING":self.ORANGE,"EVALUATING":self.YELLOW,"ERROR":self.RED}.get(st.get("status","READY"),self.WHITE)
        cv2.circle(out,(218,28),7,sc,-1,cv2.LINE_AA)
        cv2.putText(out,st.get("status","READY"),(232,34),self.F,0.5,sc,1,cv2.LINE_AA)
        cv2.putText(out,"☝ index up=DRAW  ✋ open hand=LIFT  🤏 pinch/SPACE=EVAL",(316,34),self.F,0.36,self.MUTED,1,cv2.LINE_AA)
        cv2.putText(out,f"{st.get('fps',0):.0f}fps",(self.ow-82,34),self.FP,1.1,self.MUTED,1,cv2.LINE_AA)
        cy=self.hh; roi=out[cy:cy+self.fh,:self.fw]
        if cam is not None: roi[:]=cam
        a=cbgra[:,:,3:4].astype(np.float32)/255
        roi[:]=(roi.astype(np.float32)*(1-a)+cbgra[:,:,:3].astype(np.float32)*a).astype(np.uint8)
        if st.get("skel") and st.get("lm"):
            h2,w2=roi.shape[:2]; pts=[(int(l.x*w2),int(l.y*h2)) for l in st["lm"].landmark]
            for a2,b2 in mp.solutions.hands.HAND_CONNECTIONS: cv2.line(roi,pts[a2],pts[b2],(45,45,80),1,cv2.LINE_AA)
            for p in pts: cv2.circle(roi,p,3,(80,80,150),-1,cv2.LINE_AA)
        fp=st.get("fp")
        if fp:
            fx,fy=fp; dr=st.get("drawing",False); col=self.GREEN if dr else (150,150,150)
            cv2.circle(roi,(fx,fy),24,(col[0]//3,col[1]//3,col[2]//3),2,cv2.LINE_AA)
            cv2.circle(roi,(fx,fy),11,col,-1,cv2.LINE_AA)
            cv2.circle(roi,(fx,fy),4,(255,255,255),-1,cv2.LINE_AA)
            lbl="DRAWING ✏" if dr else st.get("hint","")
            cv2.putText(roi,lbl,(fx+28,fy-10),self.F,0.48,col,1,cv2.LINE_AA)
        nf=st.get("nf",0); bc2=self.GREEN if st.get("drawing") else self.MUTED
        cv2.putText(roi,f"fingers up:{nf}",(14,self.fh-14),self.F,0.48,bc2,1,cv2.LINE_AA)
        cv2.line(out,(self.fw,cy),(self.fw,cy+self.fh),self.BORDER,1)
        self._panel(out[cy:cy+self.fh,self.fw:],st)
        f0=cy+self.fh; out[f0:,:]=self.PANEL
        cv2.line(out,(0,f0),(self.ow,f0),self.BORDER,1); self._footer(out[f0:])
        # Overlay recognized expression/result on the webcam feed area (large, centered).
        expr = st.get("expr","")
        res  = st.get("result","")
        if expr and expr != "—":
            txt = expr.replace(" ", "")
            scale = 2.2
            while cv2.getTextSize(txt,self.F,scale,3)[0][0] > self.fw-80 and scale > 0.8:
                scale -= 0.1
            txw, txh = cv2.getTextSize(txt,self.F,scale,3)[0]
            tx = int((self.fw - txw) / 2)
            ty = int(self.hh + 60 + txh)
            cv2.putText(out, txt, (tx, ty), self.F, scale, (0,0,255), 4, cv2.LINE_AA)
        if res and res != "?":
            scale = 1.6
            rw, rh = cv2.getTextSize(str(res),self.F,scale,3)[0]
            rx = int((self.fw - rw) / 2)
            ry = int(self.hh + 120 + rh)
            cv2.putText(out, str(res), (rx, ry), self.F, scale, (0,255,0), 3, cv2.LINE_AA)
        return out
    def _panel(self,roi,st):
        roi[:]=self.PANEL; h,w=roi.shape[:2]; y=16
        self._lbl(roi,"RECOGNIZED",14,y); y+=22
        cv2.putText(roi,str(st.get("expr_raw","—"))[:34],(14,y),self.F,0.55,(160,160,200),1,cv2.LINE_AA); y+=32
        self._lbl(roi,"EXPRESSION",14,y); y+=22
        cv2.putText(roi,str(st.get("expr","—"))[:34],(14,y),self.F,0.58,self.WHITE,1,cv2.LINE_AA); y+=34
        self._lbl(roi,"RESULT",14,y); y+=26
        res=str(st.get("result","?")); rcol=self.GREEN if not res.startswith("Error") else self.RED
        sc=2.4
        while cv2.getTextSize(res,self.F,sc,3)[0][0]>w-28 and sc>0.5: sc-=0.15
        _,th=cv2.getTextSize(res,self.F,sc,3)[0]
        cv2.putText(roi,res,(14,y+th),self.F,sc,rcol,3,cv2.LINE_AA); y+=th+16
        conf=st.get("conf",0.0); self._lbl(roi,f"CONFIDENCE  {int(conf*100)}%",14,y); y+=20
        bw=w-28; cv2.rectangle(roi,(14,y),(14+bw,y+8),self.BORDER,-1)
        fill=int(bw*conf)
        for i in range(fill):
            t=i/max(fill,1); cv2.line(roi,(14+i,y),(14+i,y+8),(int(255*(1-t)),int(150+105*t),int(t*80)))
        cv2.rectangle(roi,(14,y),(14+bw,y+8),self.BORDER,1); y+=26
        cv2.line(roi,(14,y),(w-14,y),self.BORDER,1); y+=14
        self._lbl(roi,"HISTORY",14,y); y+=22
        hist=st.get("history",[])
        if not hist: cv2.putText(roi,"No evaluations yet.",(14,y+14),self.F,0.4,self.MUTED,1,cv2.LINE_AA)
        for i,(e,r) in enumerate(reversed(hist[-8:])):
            a2=max(0.3,1.0-i*0.11); c=tuple(int(v*a2) for v in self.MUTED)
            cv2.putText(roi,f"{e[:22]} = {r}",(14,y+14),self.F,0.4,c,1,cv2.LINE_AA); y+=20
            if y>h-65: break
        y=h-54; cv2.line(roi,(14,y),(w-14,y),self.BORDER,1); y+=10
        bc=st.get("brush_color",(0,255,150)); bsz=st.get("brush_size",14)
        cv2.circle(roi,(28,y+14),min(bsz//2,14),bc,-1,cv2.LINE_AA)
        cv2.putText(roi,f"size {bsz}  1-5=color  +/-=size",(50,y+18),self.F,0.35,self.MUTED,1,cv2.LINE_AA)
    def _footer(self,roi):
        keys=[("SPACE","Evaluate"),("C","Clear"),("Z","Undo"),("S","Skeleton"),("1-5","Color"),("+/-","Size"),("Q","Quit")]
        x=16
        for k,d in keys:
            kw=cv2.getTextSize(k,self.F,0.38,1)[0][0]
            cv2.rectangle(roi,(x,14),(x+kw+10,34),self.BORDER,-1); cv2.rectangle(roi,(x,14),(x+kw+10,34),self.PURPLE,1)
            cv2.putText(roi,k,(x+5,29),self.F,0.38,self.PURPLE,1,cv2.LINE_AA); x+=kw+16
            dw=cv2.getTextSize(d,self.F,0.38,1)[0][0]
            cv2.putText(roi,d,(x,29),self.F,0.38,self.MUTED,1,cv2.LINE_AA); x+=dw+20
        cv2.putText(roi,"☝ index=DRAW  ✋ open=LIFT  🤏 pinch=EVAL  SPACE=evaluate anytime",
                    (16,55),self.F,0.36,self.MUTED,1,cv2.LINE_AA)
    def _lbl(self,roi,t,x,y): cv2.putText(roi,t,(x,y+10),self.F,0.34,self.MUTED,1,cv2.LINE_AA)

# ── MAIN APP ─────────────────────────────────────────────────
class App:
    def __init__(self):
        self.hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=1,
            model_complexity=1,min_detection_confidence=0.60,min_tracking_confidence=0.50)
        self.gesture=Gesture(); self.canvas=Canvas(CAM_W,CAM_H)
        self.ui=UI(CAM_W,CAM_H,PANEL_W,HEAD_H,FOOT_H)
        self.eval=Evaluator(); self.ocr=OCR()
        # Use improved OCR pipeline for better recognition
        self.ocr2 = math_ocr.MathOCR()
        self.color_idx=0; self.brush_size=BRUSH_DEF; self.was_draw=False
        self.pinch_cd=0; self.skel=True; self.history=[]; self.flash=0
        self.fps_buf=deque(maxlen=30)
        self.st=dict(expr="—",expr_raw="",result="?",conf=0.0,status="READY",fps=0,
            brush_color=COLORS[0],brush_size=BRUSH_DEF,history=self.history,skel=True,
            lm=None,fp=None,drawing=False,nf=0,hint="")
    @property
    def color(self): return COLORS[self.color_idx%len(COLORS)]
    def run(self):
        cap=cv2.VideoCapture(CAM_IDX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H); cap.set(cv2.CAP_PROP_FPS,30)
        cv2.namedWindow(WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN,self.ui.ow,self.ui.oh)
        print("\n╔══════════════════════════════════════════╗")
        print("║   AirMath v4 — Fixed Recognition!       ║")
        print("╠══════════════════════════════════════════╣")
        print("║  ☝  Only index UP   → DRAWS              ║")
        print("║  ✋  Open hand       → LIFTS pen          ║")
        print("║  🤏  Pinch / SPACE   → EVALUATE           ║")
        print("║  C=clear  Z=undo  Q=quit                  ║")
        print("╚══════════════════════════════════════════╝\n")
        fail_count=0
        while True:
            t0=time.time(); ok,frm=cap.read()
            if not ok:
                fail_count+=1
                time.sleep(0.05)
                if fail_count>=8:
                    cap.release()
                    time.sleep(0.2)
                    cap=cv2.VideoCapture(CAM_IDX)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H); cap.set(cv2.CAP_PROP_FPS,30)
                    fail_count=0
                continue
            fail_count=0
            try:
                frm=cv2.flip(frm,1)
                res=self.hands.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
                self._tick(res,frm)
                out=self.ui.frame(frm,self.canvas.render(),self.st)
            except Exception as e:
                self.st.update(status="ERROR",result=f"Runtime error: {e}",conf=0.0)
                out=self.ui.frame(frm if ok else None,self.canvas.render(),self.st)
            if self.flash>0:
                ov=out.copy(); cv2.rectangle(ov,(0,HEAD_H),(CAM_W,HEAD_H+CAM_H),(0,255,150),-1)
                cv2.addWeighted(ov,self.flash/8*0.20,out,1-self.flash/8*0.20,0,out); self.flash-=1
            cv2.imshow(WIN,out)
            self.fps_buf.append(time.time()-t0)
            self.st["fps"]=1.0/(sum(self.fps_buf)/len(self.fps_buf))
            if not self._key(cv2.waitKey(1)&0xFF): break
        cap.release(); cv2.destroyAllWindows(); print("Bye!")
    def _tick(self,results,frm):
        h,w=frm.shape[:2]; self.st.update(lm=None,fp=None,drawing=False,nf=0)
        if results.multi_hand_landmarks:
            lm=results.multi_hand_landmarks[0]; self.st["lm"]=lm
            tip=lm.landmark[IDX_TIP]; fx,fy=int(tip.x*w),int(tip.y*h)
            self.st["fp"]=(fx,fy)
            drawing,pinching,nf=self.gesture.update(lm)
            self.st.update(drawing=drawing,nf=nf,status="DRAWING" if drawing else "READY")
            self.st["hint"]="" if drawing else ("PINCH!" if pinching else f"curl fingers (up:{nf})")
            if drawing:
                if not self.was_draw: self.canvas.pen_down(self.color,self.brush_size)
                self.canvas.draw(fx,fy)
            elif self.was_draw: self.canvas.pen_up()
            self.was_draw=drawing
            if pinching and self.pinch_cd<=0: self._eval(); self.pinch_cd=55
            if self.pinch_cd>0: self.pinch_cd-=1
        else:
            if self.was_draw: self.canvas.pen_up()
            self.was_draw=False; self.st.update(hint="No hand",status="READY")
        self.st.update(brush_color=self.color,brush_size=self.brush_size,skel=self.skel)
    def _eval(self):
        if not self.canvas.has_content:
            self.st.update(status="ERROR",expr="Canvas empty!",result="Draw something",conf=0.0); return
        self.st["status"]="EVALUATING"
        try:
            raw,oconf=self.ocr2.recognize(self.canvas.to_ocr())
            print(f"  [OCR] \"{raw}\"  conf={oconf:.0%}")
            if not raw.strip():
                self.st.update(expr_raw="(nothing)",expr="(unrecognized)",result="Write clearly",conf=0.0,status="ERROR"); return
            clean,result,econf=self.eval.calc(raw)
            conf=oconf*econf
            self.st.update(expr_raw=raw,expr=clean,result=result,conf=conf,
                           status="READY" if not result.startswith("Error") else "ERROR")
            if not result.startswith("Error"): self.history.append((clean,result)); self.flash=10
            print(f"  [RESULT]  {clean} = {result}  (conf {conf:.0%})")
        except Exception as e:
            self.st.update(status="ERROR",expr_raw="(error)",expr="(error)",result=f"Error: {e}",conf=0.0)
    def _key(self,k):
        if k in (ord("q"),ord("Q"),27): return False
        if k==ord(" "): self._eval()
        elif k in (ord("c"),ord("C")): self.canvas.clear(); self.st.update(expr="—",expr_raw="",result="?",conf=0.0,status="READY")
        elif k in (ord("z"),ord("Z")): self.canvas.undo()
        elif k in (ord("s"),ord("S")): self.skel=not self.skel
        elif ord("1")<=k<=ord("5"): self.color_idx=k-ord("1")
        elif k in (ord("+"),43): self.brush_size=min(30,self.brush_size+2)
        elif k in (ord("-"),45): self.brush_size=max(4,self.brush_size-2)
        return True

if __name__=="__main__":
    App().run()
