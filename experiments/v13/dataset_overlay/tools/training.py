"""Version 1.3 adapter: unchanged images and labels, explicit time and quality.

from tools.training import TrainingDataset
ds = TrainingDataset('planetx-dataset-v1.3.0')
sample = ds.load_sequence(ds.sequences(split='train').iloc[0].stack_id)

No normalization, target recoding, balancing, boxes, or background labels are imposed.
Images have [time, height, width] shape. x is column, y is row, both zero-based.
"""
import numpy as np
import pandas as pd
from .loader import Dataset

class TrainingDataset(Dataset):
    def load(self, stack_id, qc_only=True):
        if getattr(self,'_active_stack',None)!=stack_id:
            self._tables.clear()
            self._active_stack=stack_id
        return super().load(stack_id,qc_only=qc_only)

    def _table(self, mission, name):
        key=(mission,name)
        if key not in self._tables:
            path=self.root/mission/f'{name}.parquet'
            self._tables[key]=pd.read_parquet(path,filters=[('stack_id','=',self._active_stack)]) if path.exists() else pd.DataFrame()
        return self._tables[key]

    def sequences(self, split=None, mission=None, require_multiple_times=False):
        x=pd.read_parquet(self.root/'training/sequences.parquet')
        x=x[x.usable_images]
        if split is not None: x=x[x.split==split]
        if mission is not None: x=x[x.mission==mission]
        if require_multiple_times: x=x[x.temporal_comparison_possible]
        return x.reset_index(drop=True)

    def _prepare(self, sample, stack_id, realization_id=None):
        if not len(sample['frames']): raise ValueError('Sequence has no QC-passed frames')
        fr=sample['frames']
        order=np.lexsort((fr.frame_index.to_numpy(),fr.mjd_mid.to_numpy()))
        for key in ['image','sigma','valid','flags','wave','bandwidth','psf','frame_index']:
            if key in sample: sample[key]=sample[key][order]
        fr=fr.iloc[order].reset_index(drop=True)
        fr['sequence_index']=np.arange(len(fr))
        fr['elapsed_days']=fr.mjd_mid-fr.mjd_mid.min()
        fr['gap_from_previous_frame_days']=fr.mjd_mid.diff().fillna(0)
        sample['frames']=fr
        sample['time_days']=fr.elapsed_days.to_numpy(dtype=np.float64)
        sample['time_group']=fr.time_group.to_numpy()
        sample['band_or_detector']=fr['detector' if 'detector' in fr else 'band'].to_numpy()
        sample['wavelength_um']=fr.wave_center_um.to_numpy()
        mission=stack_id.split(':')[0]
        obs=pd.read_parquet(self.root/'training'/f'{mission}_observations.parquet',filters=[('stack_id','=',stack_id)])
        obs=obs.merge(fr[['frame_index','sequence_index']],on='frame_index',validate='many_to_one')
        sample['observations']=obs.sort_values(['sequence_index','src_id']).reset_index(drop=True)
        sample['source_quality']=pd.read_parquet(self.root/'training'/f'{mission}_sources.parquet',filters=[('stack_id','=',stack_id)])
        sample['tracks']=sample['tracks'].merge(fr[['frame_index','sequence_index']],on='frame_index',validate='many_to_one')
        if realization_id is not None:
            truth=pd.read_parquet(self.root/'training'/f'{mission}_injected_truth.parquet',filters=[('realization_id','=',realization_id)])
            sample['injected_truth']=truth.merge(fr[['frame_index','sequence_index']],on='frame_index',validate='many_to_one')
            # Real-source quality was measured before injection; do not mistake it for refreshed quality.
            sample['real_source_quality_is_pre_injection']=True
        sample['unlabelled_pixels_are_verified_background']=False
        return sample

    def load_sequence(self, stack_id):
        return self._prepare(self.load(stack_id,qc_only=True),stack_id)

    def load_fixed(self, realization_id):
        row=self.index[self.index.realization_id==realization_id].iloc[0]
        return self._prepare(super().load_injected(realization_id),row.stack_id,realization_id)

    def inject_training(self, stack_id, seed, n_objects=10):
        row=self.index[(self.index.stack_id==stack_id)&(self.index.sample_type=='real_stack')].iloc[0]
        if row.split!='train': raise ValueError('Fresh augmentation is restricted to training sky groups')
        if row.mission not in ['wise','spherex']: raise ValueError('Upstream injection supports WISE and SPHEREx only')
        sample=super().inject(stack_id,seed,n_objects)
        sample=self._prepare(sample,stack_id)
        truth=sample['injected_truth'].merge(sample['frames'][['frame_index','sequence_index']],on='frame_index',validate='many_to_one')
        truth['benchmark_eligible']=truth.present & truth.in_grid & truth.masked_frac.lt(.5) & ~truth.saturated
        sample['injected_truth']=truth
        sample['real_source_quality_is_pre_injection']=True
        return sample
