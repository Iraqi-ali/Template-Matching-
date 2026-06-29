"""
Copy-Move Forgery Detection Module
====================================
Detects cloned/duplicated regions within a single document — one of the
most common forgery techniques where a region is copied and pasted elsewhere.

Uses DCT-based block matching with lexicographic sorting for efficiency.
"""

import cv2, numpy as np, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ClonePair:
    source_x: int; source_y: int; source_w: int; source_h: int
    clone_x: int; clone_y: int; clone_w: int; clone_h: int
    similarity: float; distance: float


@dataclass
class CopyMoveReport:
    has_clones: bool = False
    clone_pairs: List[ClonePair] = field(default_factory=list)
    clone_count: int = 0; confidence: float = 0.0
    annotated_image: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0
    details: List[str] = field(default_factory=list)


class CopyMoveDetector:
    def __init__(self, block_size=16, search_step=4, min_distance=50, similarity_threshold=0.95):
        self.bs=block_size; self.step=search_step; self.min_dist=min_distance; self.thresh=similarity_threshold

    def detect(self, image: np.ndarray) -> CopyMoveReport:
        t0=time.perf_counter(); report=CopyMoveReport()
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        h,w=gray.shape; bs,step=self.bs,self.step
        features,positions=[],[]
        for y in range(0,h-bs+1,step):
            for x in range(0,w-bs+1,step):
                block=gray[y:y+bs,x:x+bs]; dct=cv2.dct(np.float32(block))
                feat=dct[:8,:8].flatten(); norm=np.linalg.norm(feat)
                if norm>0:feat/=norm
                features.append(feat); positions.append((x,y))
        if len(features)<2: report.elapsed_ms=(time.perf_counter()-t0)*1000; return report
        features=np.array(features)
        sort_idx=np.lexsort(features[:,:4].T[::-1])
        sorted_f=features[sort_idx]; sorted_p=[positions[i] for i in sort_idx]
        pairs=[]
        for i in range(len(sorted_f)):
            for j in range(i+1,min(i+21,len(sorted_f))):
                diff=np.sum(np.abs(sorted_f[i]-sorted_f[j])); sim=1.0-diff/2.0
                if sim>=self.thresh:
                    x1,y1=sorted_p[i]; x2,y2=sorted_p[j]
                    dist=np.sqrt((x1-x2)**2+(y1-y2)**2)
                    if dist>=self.min_dist: pairs.append((sim,dist,x1,y1,x2,y2))
        pairs.sort(key=lambda p:p[0],reverse=True); used=set()
        for (sim,dist,x1,y1,x2,y2) in pairs[:50]:
            k1=(x1//(bs*2),y1//(bs*2)); k2=(x2//(bs*2),y2//(bs*2))
            if k1 in used or k2 in used: continue
            used.add(k1); used.add(k2)
            report.clone_pairs.append(ClonePair(source_x=x1,source_y=y1,source_w=bs,source_h=bs,clone_x=x2,clone_y=y2,clone_w=bs,clone_h=bs,similarity=sim,distance=dist))
        report.clone_count=len(report.clone_pairs); report.has_clones=report.clone_count>0
        report.confidence=min(report.clone_count*0.15,1.0)
        report.annotated_image=self._annotate(image,report)
        report.heatmap=self._heatmap(image,report)
        report.details=self._details(report)
        report.elapsed_ms=(time.perf_counter()-t0)*1000; return report

    def _annotate(self,img,report): 
        out=img.copy(); colors=[(0,0,255),(0,128,255),(0,255,128),(255,0,128),(255,128,0),(128,0,255),(255,0,255),(0,255,255)]
        for i,p in enumerate(report.clone_pairs):
            c=colors[i%8]
            cv2.rectangle(out,(p.source_x,p.source_y),(p.source_x+p.source_w,p.source_y+p.source_h),c,3)
            cv2.rectangle(out,(p.clone_x,p.clone_y),(p.clone_x+p.clone_w,p.clone_y+p.clone_h),c,3)
            cx1=p.source_x+p.source_w//2;cy1=p.source_y+p.source_h//2;cx2=p.clone_x+p.clone_w//2;cy2=p.clone_y+p.clone_h//2
            cv2.line(out,(cx1,cy1),(cx2,cy2),c,2); cv2.circle(out,(cx1,cy1),5,c,-1); cv2.circle(out,(cx2,cy2),5,c,-1)
            cv2.putText(out,f"C{i+1}",(p.clone_x,p.clone_y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,2)
        h,w=out.shape[:2]; cv2.rectangle(out,(0,0),(w,34),(0,0,0),-1)
        cv2.putText(out,f"COPY-MOVE: {report.clone_count} clone(s) | {report.confidence:.0%}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        return out

    def _heatmap(self,img,report):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img; hm=np.zeros(gray.shape,np.float32)
        for p in report.clone_pairs:
            for(x,y)in[(p.source_x,p.source_y),(p.clone_x,p.clone_y)]:
                y1=max(0,y-5);y2=min(hm.shape[0],y+p.source_h+5);x1=max(0,x-5);x2=min(hm.shape[1],x+p.source_w+5)
                hm[y1:y2,x1:x2]+=p.similarity
        if np.max(hm)>0:hm/=np.max(hm)
        return hm

    def _details(self,report):
        d=[]; 
        if report.has_clones:
            d.append(f"🔴 COPY-MOVE DETECTED: {report.clone_count} clone(s)")
            for i,p in enumerate(report.clone_pairs): d.append(f"  C{i+1}: ({p.source_x},{p.source_y})→({p.clone_x},{p.clone_y}) sim={p.similarity:.3f}")
        else: d.append("✅ No copy-move forgery detected")
        return d
