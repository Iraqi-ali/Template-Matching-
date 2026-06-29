"""
Font & Text Consistency Analyzer
=================================
Detects font/typeface inconsistencies in documents — different fonts,
sizes, weights, or digitally altered text within the same document.
"""

import cv2, numpy as np, time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FontRegion:
    x:int;y:int;width:int;height:int
    estimated_height:float; estimated_weight:float; consistency_score:float=1.0


@dataclass
class FontReport:
    is_consistent:bool=True; inconsistent_regions:List[FontRegion]=field(default_factory=list)
    avg_height:float=0.0;avg_weight:float=0.0;height_variation:float=0.0;weight_variation:float=0.0
    confidence:float=1.0;annotated_image:Optional[np.ndarray]=None;elapsed_ms:float=0.0
    details:List[str]=field(default_factory=list)


class FontAnalyzer:
    def __init__(self,min_text_height=12,max_text_height=200,height_tolerance=0.20,weight_tolerance=0.25):
        self.min_h=min_text_height;self.max_h=max_text_height;self.h_tol=height_tolerance;self.w_tol=weight_tolerance

    def analyze(self,image:np.ndarray)->FontReport:
        t0=time.perf_counter();report=FontReport()
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,3))
        closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: report.elapsed_ms=(time.perf_counter()-t0)*1000;report.details.append("No text regions found");return report
        regions=[]
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area<50:continue
            x,y,w,h=cv2.boundingRect(cnt)
            if h<self.min_h or h>self.max_h:continue
            if w<5:continue
            roi=binary[y:y+h,x:x+w];stroke_weight=np.sum(roi>0)/max(roi.shape[1],1)/255.0
            if stroke_weight>0:regions.append({'x':x,'y':y,'w':w,'h':h,'height':h,'weight':stroke_weight})
        if len(regions)<2:report.elapsed_ms=(time.perf_counter()-t0)*1000;report.details.append("Not enough text regions");return report
        heights=[r['height'] for r in regions];weights=[r['weight'] for r in regions]
        avg_h=np.mean(heights);std_h=np.std(heights);avg_w=np.mean(weights);std_w=np.std(weights)
        report.avg_height=avg_h;report.avg_weight=avg_w
        report.height_variation=std_h/avg_h if avg_h>0 else 0;report.weight_variation=std_w/avg_w if avg_w>0 else 0
        for r in regions:
            h_dev=abs(r['height']-avg_h)/avg_h if avg_h>0 else 0;w_dev=abs(r['weight']-avg_w)/avg_w if avg_w>0 else 0
            consistency=1.0-(h_dev+w_dev)/2
            if h_dev>self.h_tol or w_dev>self.w_tol:report.inconsistent_regions.append(FontRegion(x=r['x'],y=r['y'],width=r['w'],height=r['h'],estimated_height=r['height'],estimated_weight=r['weight'],consistency_score=consistency))
        report.is_consistent=len(report.inconsistent_regions)==0
        report.confidence=1.0-min(len(report.inconsistent_regions)*0.1,1.0)
        report.annotated_image=self._annotate(image,report)
        report.details=self._details(report)
        report.elapsed_ms=(time.perf_counter()-t0)*1000;return report

    def _annotate(self,img,report):
        out=img.copy()
        for i,r in enumerate(report.inconsistent_regions):
            cv2.rectangle(out,(r.x-2,r.y-2),(r.x+r.width+2,r.y+r.height+2),(0,165,255),2)
            cv2.putText(out,f"F{i+1} h={r.estimated_height:.0f}",(r.x,r.y-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,165,255),1)
        h,w=out.shape[:2];cv2.rectangle(out,(0,0),(w,34),(0,0,0),-1)
        s="FONT: Consistent" if report.is_consistent else f"FONT: {len(report.inconsistent_regions)} inconsistent"
        cv2.putText(out,s,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1);return out

    def _details(self,report):
        d=[]
        if report.is_consistent:d.append("✅ Font/text appears consistent"); 
        else:
            d.append(f"⚠️ {len(report.inconsistent_regions)} inconsistent font region(s)")
            for i,r in enumerate(report.inconsistent_regions[:10]):d.append(f"  Region {i+1}: ({r.x},{r.y}) h={r.estimated_height:.0f}px w={r.estimated_weight:.1f}")
        d.append(f"  Avg height: {report.avg_height:.1f}px (±{report.height_variation:.1%})")
        d.append(f"  Avg weight: {report.avg_weight:.1f} (±{report.weight_variation:.1%})");return d
