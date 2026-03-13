import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

# Target Classes from TransientHunter
TARGET_CLASSES = [
    "SN Ia", "SN II", "AGN", "QSO", "RRLyrae", 
    "Cepheid", "EB", "LPV", "CV", "Blazar"
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(TARGET_CLASSES)}

class LightCurvePreprocessor:
    """Preprocess astronomical light curves for deep learning"""
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.feature_scaler = None
        self.sequence_stats = None
        
    def load_metadata(self, metadata_path: Path):
        """Load normalization stats if available"""
        if metadata_path.exists():
            import pickle
            with open(metadata_path, 'rb') as f:
                meta = pickle.load(f)
                self.feature_scaler = meta.get('scaler')
                self.sequence_stats = meta.get('raw_sequence_stats')

    def extract_features(self, lc: pd.DataFrame) -> np.ndarray:
        """Extract statistical features from light curve matching training"""
        features = {}
        
        for band, fid in [('g', 1), ('r', 2)]:
            band_data = lc[lc['fid'] == fid]
            
            if len(band_data) > 0:
                mag = band_data['magpsf'].values
                err = band_data['sigmapsf'].values
                mjd = band_data['mjd'].values
                
                features[f'mean_mag_{band}'] = np.mean(mag)
                features[f'std_mag_{band}'] = np.std(mag) if len(mag) > 1 else 0
                features[f'median_mag_{band}'] = np.median(mag)
                features[f'min_mag_{band}'] = np.min(mag)
                features[f'max_mag_{band}'] = np.max(mag)
                features[f'amplitude_{band}'] = np.max(mag) - np.min(mag)
                features[f'mag_10pct_{band}'] = np.percentile(mag, 10)
                features[f'mag_90pct_{band}'] = np.percentile(mag, 90)
                features[f'mad_{band}'] = np.median(np.abs(mag - np.median(mag)))
                features[f'iqr_{band}'] = np.percentile(mag, 75) - np.percentile(mag, 25)
                
                weights = 1.0 / (err ** 2 + 1e-10)
                features[f'wmean_{band}'] = np.average(mag, weights=weights)
                
                if len(mag) > 3:
                    features[f'skew_{band}'] = pd.Series(mag).skew()
                    features[f'kurtosis_{band}'] = pd.Series(mag).kurtosis()
                else:
                    features[f'skew_{band}'] = 0
                    features[f'kurtosis_{band}'] = 0
                
                if len(mjd) > 1:
                    dt = np.diff(mjd)
                    features[f'mean_dt_{band}'] = np.mean(dt)
                    features[f'std_dt_{band}'] = np.std(dt)
                    dmag_dt = np.diff(mag) / (np.diff(mjd) + 1e-10)
                    features[f'mean_dmag_dt_{band}'] = np.mean(np.abs(dmag_dt))
                    features[f'max_dmag_dt_{band}'] = np.max(np.abs(dmag_dt))
                else:
                    features[f'mean_dt_{band}'] = 0
                    features[f'std_dt_{band}'] = 0
                    features[f'mean_dmag_dt_{band}'] = 0
                    features[f'max_dmag_dt_{band}'] = 0
            else:
                for key in ['mean_mag', 'std_mag', 'median_mag', 'min_mag', 'max_mag',
                           'amplitude', 'mag_10pct', 'mag_90pct', 'mad', 'iqr',
                           'wmean', 'skew', 'kurtosis', 'mean_dt', 'std_dt',
                           'mean_dmag_dt', 'max_dmag_dt']:
                    features[f'{key}_{band}'] = 0
        
        if features.get('mean_mag_g', 0) != 0 and features.get('mean_mag_r', 0) != 0:
            features['mean_color'] = features['mean_mag_g'] - features['mean_mag_r']
            features['amplitude_ratio'] = features.get('amplitude_g', 0) / (features.get('amplitude_r', 1e-10) + 1e-10)
        else:
            features['mean_color'] = 0
            features['amplitude_ratio'] = 1.0
        
        features['n_obs_total'] = len(lc)
        features['n_obs_g'] = len(lc[lc['fid'] == 1])
        features['n_obs_r'] = len(lc[lc['fid'] == 2])
        features['duration'] = lc['mjd'].max() - lc['mjd'].min()
        
        # Order exactly matching training (40 features)
        feature_order = [
            'mean_mag_g', 'std_mag_g', 'median_mag_g', 'min_mag_g', 'max_mag_g',
            'amplitude_g', 'mag_10pct_g', 'mag_90pct_g', 'mad_g', 'iqr_g',
            'wmean_g', 'skew_g', 'kurtosis_g', 'mean_dt_g', 'std_dt_g',
            'mean_dmag_dt_g', 'max_dmag_dt_g',
            'mean_mag_r', 'std_mag_r', 'median_mag_r', 'min_mag_r', 'max_mag_r',
            'amplitude_r', 'mag_10pct_r', 'mag_90pct_r', 'mad_r', 'iqr_r',
            'wmean_r', 'skew_r', 'kurtosis_r', 'mean_dt_r', 'std_dt_r',
            'mean_dmag_dt_r', 'max_dmag_dt_r',
            'mean_color', 'amplitude_ratio', 'n_obs_total', 'n_obs_g', 'n_obs_r', 'duration'
        ]
        
        result = np.zeros(40, dtype=np.float32)
        for i, name in enumerate(feature_order[:40]):
            result[i] = features.get(name, 0)
        
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def create_sequence(self, lc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create fixed-length sequences from light curve"""
        sequence = np.zeros((self.sequence_length, 5), dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)
        
        mjd = lc['mjd'].values
        if len(mjd) == 0:
            return sequence, mask
            
        unique_mjd = np.unique(mjd)
        
        if len(unique_mjd) >= self.sequence_length:
            indices = np.linspace(0, len(unique_mjd)-1, self.sequence_length, dtype=int)
            selected_mjd = unique_mjd[indices]
        else:
            selected_mjd = unique_mjd
            
        for i, t in enumerate(selected_mjd):
            if i >= self.sequence_length: break
            
            obs = lc[lc['mjd'] == t]
            t_norm = (t - mjd.min()) / (mjd.max() - mjd.min() + 1e-10)
            sequence[i, 0] = t_norm
            
            g_obs = obs[obs['fid'] == 1]
            if len(g_obs) > 0:
                sequence[i, 1] = g_obs['magpsf'].values[0]
                sequence[i, 2] = g_obs['sigmapsf'].values[0]
                
            r_obs = obs[obs['fid'] == 2]
            if len(r_obs) > 0:
                sequence[i, 3] = r_obs['magpsf'].values[0]
                sequence[i, 4] = r_obs['sigmapsf'].values[0]
                
            mask[i] = True
            
        for col in [1, 3]:
            valid_idx = np.where((sequence[:, col] != 0) & mask)[0]
            if len(valid_idx) > 0:
                for i in range(self.sequence_length):
                    if mask[i] and sequence[i, col] == 0:
                        nearest_idx = valid_idx[np.argmin(np.abs(valid_idx - i))]
                        sequence[i, col] = sequence[nearest_idx, col]
                        sequence[i, col+1] = sequence[nearest_idx, col+1]
                        
        return sequence, mask

    def prepare_input(self, observations: List[Dict]) -> Dict:
        """Prepare raw dictionary observations for model input"""
        data = {
            'mjd': [float(o['time']) for o in observations],
            'magpsf': [float(o['magnitude']) for o in observations],
            'sigmapsf': [float(o['error']) for o in observations],
            'fid': [1 if str(o['band']).lower() == 'g' else 2 for o in observations]
        }
        lc_df = pd.DataFrame(data)
        lc_df = lc_df.sort_values('mjd').reset_index(drop=True)
        
        sequence, mask = self.create_sequence(lc_df)
        features = self.extract_features(lc_df)
        
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1))[0]
        
        if self.sequence_stats:
            for col_idx in range(5):
                mean_val = self.sequence_stats.get(f'mean_{col_idx}', 0.0)
                std_val = self.sequence_stats.get(f'std_{col_idx}', 1.0)
                if std_val > 0:
                    if col_idx == 0:
                        sequence[:, col_idx] = (sequence[:, col_idx] - mean_val) / std_val
                    else:
                        sequence[:, col_idx] = np.where(
                            sequence[:, col_idx] != 0,
                            (sequence[:, col_idx] - mean_val) / std_val,
                            0.0
                        )
                        
        return {
            'sequence': sequence,
            'mask': mask,
            'features': features
        }
