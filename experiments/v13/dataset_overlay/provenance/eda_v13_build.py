import sys, json, shutil, hashlib, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'deps'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'planetx-dataset-v1.2.0/planetx-dataset-v1.2.0'
OUT=BASE/'planetx-dataset-v1.3.0'
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(df,name):
    df.to_csv(OUT/'eda'/f'{name}.csv',index=False)
def main():
    if not OUT.exists(): shutil.copytree(SRC,OUT)
    (OUT/'eda').mkdir(exist_ok=True); (OUT/'training').mkdir(exist_ok=True)
    checks=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=str(detail)))
        if not ok: print('CHECK FAILED',name,detail,flush=True)
    original=[]
    for line in (SRC/'MANIFEST.sha256').read_text().splitlines():
        h,p=line.split(None,1); p=p.strip()
        actual=digest(OUT/p); check('upstream hash '+p,actual==h)
        original.append(dict(path=p,sha256=h))
    print('Copied and verified original files',flush=True)
    pd.DataFrame(original).to_csv(OUT/'eda/upstream_files.csv',index=False)
    idx=pd.read_parquet(OUT/'index.parquet'); real=idx[idx.sample_type=='real_stack'].copy()
    check('each sky group stays in one split',idx.groupby('group').split.nunique().max()==1)
    for r in idx.itertuples(): check('indexed path '+r.path,(OUT/r.path).is_file())
    schemas=[]; totals=[]; numbers=[]
    for p in SRC.rglob('*.parquet'):
        df=pd.read_parquet(p); rel=p.relative_to(SRC).as_posix()
        totals.append(dict(file=rel,rows=len(df),columns=len(df.columns)))
        for c in df:
            schemas.append(dict(file=rel,column=c,dtype=str(df[c].dtype),rows=len(df),nulls=int(df[c].isna().sum())))
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                a=pd.to_numeric(df[c],errors='coerce').to_numpy(dtype=float); a=a[np.isfinite(a)]
                if len(a): numbers.append(dict(file=rel,column=c,n_finite=len(a),minimum=float(a.min()),p01=float(np.quantile(a,.01)),median=float(np.median(a)),p99=float(np.quantile(a,.99)),maximum=float(a.max())))
    save(pd.DataFrame(totals),'table_inventory'); save(pd.DataFrame(schemas),'column_missingness'); save(pd.DataFrame(numbers),'numeric_distributions')
    allf=[]; alls=[]; seqs=[]; classes=[]; synth=[]; obsstats=[]; image_stats=[]; gallery=[]
    for mission in ['wise','spherex','comet','neossat']:
        print('Scanning',mission,flush=True)
        f=pd.read_parquet(OUT/mission/'frames.parquet'); s=pd.read_parquet(OUT/mission/'sources.parquet'); t=pd.read_parquet(OUT/mission/'tracks.parquet')
        meta=real[real.mission==mission].set_index('stack_id')
        for d in [f,s]: d['split']=d.stack_id.map(meta.split); d['mission']=mission
        check(mission+' frame primary key',not f.duplicated(['stack_id','frame_index']).any())
        check(mission+' source primary key',not s.duplicated(['stack_id','src_id']).any())
        check(mission+' track primary key',not t.duplicated(['stack_id','src_id','frame_index']).any())
        tf=t.merge(f[['stack_id','frame_index','qc_pass','mjd_mid','time_group','split']],on=['stack_id','frame_index'],how='left',validate='many_to_one',indicator=True)
        check(mission+' all track frame joins',tf['_merge'].eq('both').all()); tf=tf.drop(columns='_merge')
        ts=tf.merge(s[['stack_id','src_id','class','is_mover','mover_detectable']],on=['stack_id','src_id'],how='left',validate='many_to_one',indicator=True)
        edge=255 if mission=='neossat' else 127
        eligible_join=ts.qc_pass & ts.x.between(0,edge) & ts.y.between(0,edge)
        check(mission+' usable in-image tracks have source labels',ts.loc[eligible_join,'_merge'].eq('both').all(),f'{int(ts._merge.eq("left_only").sum())} unmatched rows outside usable image domain; retained with null source label')
        ts=ts.drop(columns='_merge')
        ing=ts.x.between(0,edge)&ts.y.between(0,edge); solar=ts.src_kind.eq('sso')
        eff=np.where(solar & np.isfinite(ts.snr_diff),ts.snr_diff,ts.snr)
        off=np.hypot(ts.centroid_dx,ts.centroid_dy)
        clean=np.where(solar,~ts.blended.astype(bool)&~ts.artifact_vetoes_mover,~ts.contaminated.astype(bool)&~(ts.saturated&~ts.saturated_self)) & ~(off>1.5)
        detected=ts.qc_pass & ing & (np.nan_to_num(eff)>=5) & ts.aperture_valid_frac.ge(.95) & clean
        visible=detected | (ts.qc_pass & ing & ts.saturated_self & ~solar)
        flags=ts[['stack_id','src_id','frame_index','x','y','class','is_mover','mover_detectable','time_group','mjd_mid','split']].copy()
        flags['qc_pass']=ts.qc_pass; flags['in_grid']=ing; flags['catalog_detected']=detected; flags['catalog_visible']=visible
        flags['saturated']=ts.saturated; flags['contaminated']=ts.contaminated; flags['blended']=ts.blended
        flags['effective_snr']=eff
        flags['label_status']=np.select([~ts.qc_pass,~ing,visible],['failed_frame','outside_image','visible_catalog_source'],default='uncertain_catalog_source')
        flags.to_parquet(OUT/'training'/f'{mission}_observations.parquet',index=False)
        epochs=flags[visible].groupby(['stack_id','src_id']).time_group.nunique().rename('visible_distinct_times')
        s=s.merge(epochs,on=['stack_id','src_id'],how='left'); s.visible_distinct_times=s.visible_distinct_times.fillna(0).astype(int)
        s['motion_observable_two_times']=s.is_mover & s.visible_distinct_times.ge(2)
        s.to_parquet(OUT/'training'/f'{mission}_sources.parquet',index=False)
        for split,g in s.groupby('split'):
            for c,gg in g.groupby('class'):
                classes.append(dict(mission=mission,split=split,object_class=c,stack_source_rows=len(gg),percent_within_mission_split=100*len(gg)/len(g),visible_any=int(gg.n_frames_visible.gt(0).sum()),movers_two_distinct_times=int(gg.motion_observable_two_times.sum())))
        for (split,status),g in flags.groupby(['split','label_status']): obsstats.append(dict(mission=mission,split=split,status=status,rows=len(g)))
        for r in meta.reset_index().itertuples():
            ff=f[f.stack_id==r.stack_id].sort_values(['mjd_mid','frame_index']); good=ff[ff.qc_pass]
            unique=np.sort(good.groupby('time_group').mjd_mid.mean().to_numpy()); gaps=np.diff(unique)
            seqs.append(dict(stack_id=r.stack_id,mission=mission,split=r.split,group=r.group,path=r.path,n_frames=len(ff),n_qc=len(good),distinct_times=len(unique),time_span_days=float(np.ptp(unique)) if len(unique) else 0,median_gap_days=float(np.median(gaps)) if len(gaps) else np.nan,min_gap_days=float(gaps.min()) if len(gaps) else np.nan,max_gap_days=float(gaps.max()) if len(gaps) else np.nan,usable_images=len(good)>0,temporal_comparison_possible=len(unique)>=2))
            with np.load(OUT/r.path,allow_pickle=False) as archive:
                z={k:archive[k] for k in archive.files}
                check(r.stack_id+' image rows match metadata',len(z['image'])==len(ff))
                canonical=ff.sort_values('frame_index')
                if 'frame_uid' in z: check(r.stack_id+' image uid order',np.array_equal(z['frame_uid'].astype(str),canonical.frame_uid.to_numpy().astype(str)))
                for k in ['sigma','valid','flags','wave','bandwidth']: check(r.stack_id+' '+k+' shape',z[k].shape==z['image'].shape)
                for i,row in enumerate(canonical.itertuples()):
                    a=z['image'][i]; v=z['valid'][i].astype(bool); sig=z['sigma'][i]
                    check(row.frame_uid+' valid finite image',np.isfinite(a[v]).all())
                    check(row.frame_uid+' valid positive sigma',(np.isfinite(sig[v])&(sig[v]>0)).all())
                    av=a[v & np.isfinite(a)]
                    image_stats.append(dict(mission=mission,stack_id=r.stack_id,frame_index=row.frame_index,split=r.split,qc_pass=row.qc_pass,height=a.shape[0],width=a.shape[1],valid_fraction=float(v.mean()),pixel_p01=float(np.quantile(av,.01)) if len(av) else np.nan,pixel_median=float(np.median(av)) if len(av) else np.nan,pixel_p99=float(np.quantile(av,.99)) if len(av) else np.nan))
                # Representative source-class examples; labels can be faint. Selection uses existing detection rules.
                for c in sorted(s[s.stack_id==r.stack_id]['class'].unique()):
                    if any(x['class']==c for x in gallery): continue
                    candidates=flags[(flags.stack_id==r.stack_id)&(flags['class']==c)&flags.catalog_visible]
                    if candidates.empty: continue
                    arow=candidates.sort_values('effective_snr',ascending=False).iloc[0]
                    i=int(arow.frame_index); a=z['image'][i]; v=z['valid'][i].astype(bool)
                    x,y=int(round(arow.x)),int(round(arow.y)); lo,hi=np.quantile(a[v],[.05,.995]); scale=np.arcsinh(10*np.clip((a-lo)/max(hi-lo,1e-10),0,1))/np.arcsinh(10)
                    im=Image.fromarray(np.uint8(np.where(v,scale,0)*255)).convert('RGB').resize((384,384))
                    dr=ImageDraw.Draw(im); k=384/a.shape[0]; dr.ellipse((x*k-10,y*k-10,x*k+10,y*k+10),outline='red',width=2)
                    filename=f'example_{c}.png'; im.save(OUT/'eda'/filename)
                    gallery.append(dict(class_=c,**{'class':c},stack_id=r.stack_id,src_id=arow.src_id,frame_index=i,image=filename))
        f=f.sort_values(['stack_id','mjd_mid','frame_index'])
        f['elapsed_days']=f.mjd_mid-f.groupby('stack_id').mjd_mid.transform('min')
        f['gap_from_previous_frame_days']=f.groupby('stack_id').mjd_mid.diff().fillna(0)
        f.to_parquet(OUT/'training'/f'{mission}_frames.parquet',index=False)
        allf.append(f); alls.append(s)
        if mission in ['wise','spherex']:
            o=pd.read_parquet(OUT/mission/'injected_objects.parquet'); tr=pd.read_parquet(OUT/mission/'injected_truth.parquet')
            oi=idx[(idx.mission==mission)&(idx.sample_type=='injected')][['realization_id','split']]
            o=o.merge(oi,on='realization_id',validate='many_to_one'); tr=tr.merge(o[['realization_id','obj_id','class','is_mover','split']],on=['realization_id','obj_id'],how='left',validate='many_to_one',indicator=True)
            check(mission+' injected truth object join',tr['_merge'].eq('both').all()); tr=tr.drop(columns='_merge')
            check(mission+' injected truth primary key',not tr.duplicated(['realization_id','obj_id','frame_index']).any())
            tr['benchmark_eligible']=tr.present & tr.in_grid & tr.masked_frac.lt(.5) & ~tr.saturated
            tr['benchmark_change_target']=tr.is_mover | tr['class'].isin(['transient','variable'])
            tr.to_parquet(OUT/'training'/f'{mission}_injected_truth.parquet',index=False)
            for (split,c),g in o.groupby(['split','class']):
                tt=tr[(tr.split==split)&(tr['class']==c)]
                synth.append(dict(mission=mission,split=split,object_class=c,objects=len(g),truth_rows=len(tt),eligible_rows=int(tt.benchmark_eligible.sum()),saturated_rows=int(tt.saturated.sum()),eligible_snr_below5=int((tt.benchmark_eligible & tt.snr_true.lt(5)).sum())))
    seq=pd.DataFrame(seqs); seq.to_parquet(OUT/'training/sequences.parquet',index=False); save(seq,'sequences')
    save(pd.DataFrame(classes),'class_distribution'); save(pd.DataFrame(synth),'synthetic_distribution'); save(pd.DataFrame(obsstats),'label_status'); save(pd.DataFrame(image_stats),'image_statistics'); save(pd.DataFrame(gallery).drop(columns='class_'),'example_index')
    f=pd.concat(allf,ignore_index=True); s=pd.concat(alls,ignore_index=True)
    s['motion_bin']=pd.cut(s.motion_px,[-np.inf,.1,1,5,np.inf],labels=['under_0.1px','0.1_to_1px','1_to_5px','over_5px'],right=False)
    save(s.groupby(['mission','split','motion_bin'],observed=True).agg(source_rows=('src_id','size'),visible_two_times=('visible_distinct_times',lambda x:int(x.ge(2).sum())),original_detectable_movers=('mover_detectable','sum'),movers_two_distinct_times=('motion_observable_two_times','sum')).reset_index(),'motion_coverage')
    save(f.groupby(['mission','split','qc_pass','qc_reasons'],dropna=False).size().rename('frames').reset_index(),'quality_reasons')
    save(f.groupby(['mission','split']).agg(frames=('frame_index','size'),passing=('qc_pass','sum'),median_valid_fraction=('valid_frac','median')).reset_index(),'split_summary')
    shared=f.dropna(subset=['sha256']).groupby('sha256').split.nunique(); check('source image hashes do not cross splits',not shared.gt(1).any())
    valid_sources=s.dropna(subset=['src_id']).groupby('src_id').split.nunique(); check('source ids do not cross splits',not valid_sources.gt(1).any())
    # Scan every NPZ, including coadds, PSFs and injected files, and verify fixed injection order.
    npzs=[]
    for p in OUT.rglob('*.npz'):
        with np.load(p,allow_pickle=False) as z:
            for k in z.files:
                a=z[k]; npzs.append(dict(path=p.relative_to(OUT).as_posix(),array=k,shape=str(a.shape),dtype=str(a.dtype),nonfinite=int((~np.isfinite(a)).sum()) if a.dtype.kind in 'fc' else 0))
            if p.parent.name=='injected':
                row=idx[idx.path==p.relative_to(OUT).as_posix()].iloc[0]; ff=f[(f.stack_id==row.stack_id)&f.qc_pass].sort_values('frame_index')
                check(str(p.relative_to(OUT))+' frame order',np.array_equal(z['frame_index'],ff.frame_index.to_numpy()))
                check(str(p.relative_to(OUT))+' finite valid image',np.isfinite(z['image'][z['valid'].astype(bool)]).all())
    save(pd.DataFrame(npzs),'array_inventory'); save(pd.DataFrame(checks),'validation_checks')
    summary=dict(version='1.3.0',upstream='1.2.0',real_frames=len(f),qc_frames=int(f.qc_pass.sum()),source_rows=len(s),track_rows=sum(x['rows'] for x in totals if x['file'].endswith('/tracks.parquet')),checks=len(checks),failed_checks=sum(not x['passed'] for x in checks),nonempty_sequences=int(seq.usable_images.sum()),empty_sequences=int((~seq.usable_images).sum()),two_time_sequences=int(seq.temporal_comparison_possible.sum()),original_files_verified=len(original),original_pixels_labels_splits_preserved=True)
    (OUT/'eda/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if summary['failed_checks']: raise RuntimeError('Review failed validations before publishing')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Reproduce the EDA and derived tables without changing the source package.')
    parser.add_argument('--source',type=Path,default=SRC)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); SRC=args.source.resolve(); OUT=args.output.resolve()
    if SRC==OUT or SRC in OUT.parents or OUT in SRC.parents: raise ValueError('Source and output must be separate directories')
    main()
