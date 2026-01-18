import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.core import LightningDataModule
from multiprocessing import get_context
from do_novo.utils import scale_intensity, scale_to_unit_norm, seq_to_token, make_theoretical_by_ions, build_grid


class DenovoDataModule(LightningDataModule):
    def __init__(
            self,
            cfg_dataset,
            cfg_dataloader,
            tokenizer
    ):
        super().__init__()
        self.train_file_path = cfg_dataset.train_file_path
        self.val_file_path = cfg_dataset.val_file_path
        self.test_file_path = cfg_dataset.test_file_path
        self.predict_file_path = cfg_dataset.predict_file_path
        self.cfg_dataloader = cfg_dataloader
        self.tokenizer = tokenizer


    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            self.train_ds = DenovoDataset(self.train_file_path, self.tokenizer)
            self.val_ds = DenovoDataset(self.val_file_path, self.tokenizer)

        if stage in (None, "test"):
            self.test_ds = DenovoDataset(self.test_file_path, self.tokenizer)

        if stage in (None, "predict"):
            self.predict_ds = DenovoDataset(self.predict_file_path, self.tokenizer)


    def train_dataloader(self):
        multiprocessing_context = get_context("spawn") if self.cfg_dataloader.num_workers != 0 else None
        return DataLoader(
            dataset=self.train_ds,
            shuffle=True,
            collate_fn=self.train_ds.collate_fn,
            multiprocessing_context=multiprocessing_context,
            **self.cfg_dataloader,
        )

    def val_dataloader(self):
        multiprocessing_context = get_context("spawn") if self.cfg_dataloader.num_workers != 0 else None
        return DataLoader(
            dataset=self.val_ds,
            shuffle=False,
            collate_fn=self.val_ds.collate_fn,
            multiprocessing_context=multiprocessing_context,
            **self.cfg_dataloader,
        )

    def test_dataloader(self):
        multiprocessing_context = get_context("spawn") if self.cfg_dataloader.num_workers != 0 else None
        return DataLoader(
            dataset=self.test_ds,
            shuffle=False,
            collate_fn=self.test_ds.collate_fn,
            multiprocessing_context=multiprocessing_context,
            **self.cfg_dataloader,
        )

    def predict_dataloader(self):
        multiprocessing_context = get_context("spawn") if self.cfg_dataloader.num_workers != 0 else None
        return DataLoader(
            dataset=self.predict_ds,
            shuffle=False,
            collate_fn=self.predict_ds.collate_fn_predict,
            multiprocessing_context=multiprocessing_context,
            **self.cfg_dataloader,
        )


class DenovoDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer

        if 'processed' in file_path:
            self.spec_df = pd.read_parquet(file_path)
        else:
            self.spec_df = self._get_spec_df(file_path)


        self.has_valid_label = ('label' in self.spec_df.columns) and self.spec_df['label'].notna().any()
        if self.has_valid_label:
            unique_peps = self.spec_df['label'].astype(str).unique()
            self.peptide2id = {pep: idx for idx, pep in enumerate(unique_peps)}
            self.spec_df['peptide_id'] = self.spec_df['label'].astype(str).map(self.peptide2id.get).astype(np.int64)
        else:
            self.peptide2id = {}
            self.spec_df['peptide_id'] = -1




    def __len__(self):
        return len(self.spec_df)

    def __getitem__(self, idx):
        return self.spec_df.iloc[idx]


    def _get_spec_df(self, file_path: str) -> pd.DataFrame:
        PROTON = 1.007276
        df = pd.read_parquet(file_path)

        is_predict = 'predict' in file_path

        if 'speclib' in file_path:
            ms2_mz  = df['mzarray'].map(np.hstack)
            ms2_int = df['intarray'].map(np.hstack)
            ms2_int = ms2_int.map(scale_intensity).map(scale_to_unit_norm)
            ms1_mz  = df['selected_ms1_mz'].map(np.hstack)
            ms1_int = df['selected_ms1_intensity'].map(np.hstack)
            ms1_int = ms1_int.map(scale_intensity).map(scale_to_unit_norm)

            ms2_cnts = df['mzarray'].map(lambda x: [len(a) for a in x])
            ms1_cnts = df['selected_ms1_mz'].map(lambda x: [len(a) for a in x])

            fragment_label = df['matched_ion'].map(lambda x: np.hstack(x).astype(bool).astype(int)) if not is_predict else None

        elif 'ump' in file_path:
            if df['mzarray'][0].size < 10:
                ms2_mz  = df['mzarray'].map(np.hstack)
                ms2_int = df['intarray'].map(np.hstack)
                ms2_int = ms2_int.map(scale_intensity).map(scale_to_unit_norm)
                ms1_mz  = df['selected_ms1_mz'].map(np.hstack)
                ms1_int = df['selected_ms1_intensity'].map(np.hstack)
                ms1_int = ms1_int.map(scale_intensity).map(scale_to_unit_norm)

                ms2_cnts = df['mzarray'].map(lambda x: [len(a) for a in x])
                ms1_cnts = df['selected_ms1_mz'].map(lambda x: [len(a) for a in x])

                fragment_label = df['matched_ion'].map(lambda x: np.hstack(x).astype(bool).astype(int)) if not is_predict else None
            else:
                ms2_mz  = df['mzarray']
                ms2_int = df['intarray']
                ms2_int = ms2_int.map(scale_intensity).map(scale_to_unit_norm)
                ms1_mz  = df['selected_ms1_mz']
                ms1_int = df['selected_ms1_intensity']
                ms1_int = ms1_int.map(scale_intensity).map(scale_to_unit_norm)

                ms2_cnts = df['mzarray'].map(len)
                ms1_cnts = df['selected_ms1_mz'].map(len)

                fragment_label = df['matched_ion'] if not is_predict else None
                df['rt_diff'] = 0

        else:
            raise f'There is no type in file name: {file_path}'
        
        ms2_rt = [np.repeat(rtd, cnts) for rtd, cnts in zip(df['rt_diff'], ms2_cnts)]
        ms1_rt = [np.repeat(rtd, cnts) for rtd, cnts in zip(df['rt_diff'], ms1_cnts)]

        lvl2 = ms2_mz.map(lambda a: np.full(a.size, 2, dtype=int))
        lvl1 = ms1_mz.map(lambda a: np.full(a.size, 1, dtype=int))

        mz        = [np.concatenate((a, b)) for a, b in zip(ms2_mz,  ms1_mz)]
        intensity = [np.concatenate((a, b)) for a, b in zip(ms2_int, ms1_int)]
        rt        = [np.concatenate((a, b)) for a, b in zip(ms2_rt,  ms1_rt)]
        level     = [np.concatenate((a, b)) for a, b in zip(lvl2,     lvl1)]

        target_mz       = df['window_target']
        window_upper    = df['window_upper']
        window_lower    = df['window_lower']
        charge          = df['z']
        label           = df['label']
        global_rt       = df['rt']
        frac            = df['frac'] if not is_predict else pd.Series([0.0]*len(df))
        mz1             = df['mz1']

        if (not is_predict) and ('unmatched_ion' in df.columns):
            unmatched_mz = df['unmatched_ion'].apply(
                lambda x: x + x*np.random.uniform(-20e-6, 20e-6, size=x.shape)
            ).tolist()
            unmatched_200_300_mz = [mz_[(200.0 <= mz_) & (mz_ <= 300)] for mz_ in unmatched_mz]
        else:
            unmatched_mz = [np.array([], dtype=np.float32) for _ in range(len(df))]
            unmatched_200_300_mz = [np.array([], dtype=np.float32) for _ in range(len(df))]

        if (not is_predict) and ('label' in df.columns) and df['label'].notna().any():
            theo_full_mz = [make_theoretical_by_ions(peptide)[0] for peptide in df['label']]
            theo_200_300_mz = [mz_[(200.0 <= mz_) & (mz_ <= 300)] for mz_ in theo_full_mz]
        else:
            theo_full_mz = [np.array([], dtype=np.float32) for _ in range(len(df))]
            theo_200_300_mz = [np.array([], dtype=np.float32) for _ in range(len(df))]

        spec_df = pd.DataFrame({
            'mz': mz,
            'intensity': intensity,
            'rt': rt,
            'level': level,
            'window_upper': window_upper,
            'window_lower': window_lower,
            'precursor_mass': target_mz,
            'precursor_charge': charge,
            'label': label,
            'global_rt': global_rt,
            'frac': frac,
            'fragment_label': fragment_label,
            'missing_mz': unmatched_mz, # 200~2700 (3300,3600,3800)
            'unmatched_200_300_mz': unmatched_200_300_mz,
            'theo_full_mz': theo_full_mz,
            'theo_200_300_mz': theo_200_300_mz,
            'mz1': mz1,
        })

        spec_df.to_parquet(f"{file_path.split('.')[0]}_processed.parquet", index=False)
        return spec_df


    def collate_fn(self, batch):
        model_type = torch.float32

        spectra_ls = []
        precursor_masses_ls = []
        precursor_charges_ls = []
        frac_ls = []
        peptide_seq_ls = []
        fragment_label_ls = []
        peptide_ids_ls = []
        theo_mz_bin_ls = []
        missing_mz_bin_ls = []
        window_upper_ls = []
        window_lower_ls = []
        mz1_ls = []

        for spec in batch:
            spec_tensor = torch.tensor(np.vstack(spec[['mz','intensity','rt','level']].to_numpy()).T)
            spectra_ls.append(spec_tensor)

            precursor_masses_ls.append(spec['precursor_mass'])
            precursor_charges_ls.append(spec['precursor_charge'])
            frac_ls.append(spec['frac'])
            peptide_seq_ls.append(spec['label'])
            fragment_label_ls.append(torch.from_numpy(spec['fragment_label']))
            peptide_ids_ls.append(int(spec['peptide_id']))
            
            theo_mz_bin_np = build_grid(spec['theo_200_300_mz'], 200, 300, 50)
            theo_mz_bin_ls.append(theo_mz_bin_np)
            missing_mz_bin_np = build_grid(spec['unmatched_200_300_mz'], 200, 300, 50)
            missing_mz_bin_ls.append(missing_mz_bin_np)

            window_upper_ls.append(spec['window_upper'])
            window_lower_ls.append(spec['window_lower'])
            mz1_ls.append(spec['mz1'])

        spectra = rnn.pad_sequence(spectra_ls, batch_first=True).type(model_type)
        precursors = torch.tensor(list(zip(precursor_masses_ls, precursor_charges_ls)), dtype=model_type)
        frac = torch.tensor(frac_ls).type(model_type)
        fragment_label = rnn.pad_sequence(fragment_label_ls, batch_first=True).type(model_type)
        peptide_seq = peptide_seq_ls
        peptide_token = self.tokenizer.tokenize(peptide_seq_ls, add_start=True, add_stop=True)
        peptide_ids = torch.tensor(peptide_ids_ls, dtype=torch.long)

        theo_mz_bin = torch.tensor(np.array(theo_mz_bin_ls)).type(model_type)
        missing_mz_bin = torch.tensor(np.array(missing_mz_bin_ls)).type(model_type)
        window_upper = torch.tensor(window_upper_ls, dtype=torch.long)
        window_lower = torch.tensor(window_lower_ls, dtype=torch.long)
        mz1 = torch.tensor(mz1_ls, dtype=torch.float)
        
        return spectra, precursors, peptide_token, frac, fragment_label, theo_mz_bin, missing_mz_bin, peptide_ids, window_upper, window_lower, mz1, peptide_seq



    def collate_fn_predict(self, batch):
        model_type = torch.float32

        spectra_ls = []
        precursor_masses_ls = []
        precursor_charges_ls = []
        frac_ls = []
        window_upper_ls = []
        window_lower_ls = []
        mz1_ls = []
        scan_id_ls = []

        for spec in batch:
            spec_tensor = torch.tensor(np.vstack(spec[['mz','intensity','rt','level']].to_numpy()).T)
            spectra_ls.append(spec_tensor)

            precursor_masses_ls.append(spec.get('precursor_mass', spec.get('window_target')))
            precursor_charges_ls.append(spec.get('precursor_charge', spec.get('z', 0)))
            frac_ls.append(spec.get('frac', 0))

            window_upper_ls.append(spec['window_upper'])
            window_lower_ls.append(spec['window_lower'])
            mz1_ls.append(spec.get('mz1', 0.0))

            if 'scan_id' in spec:
                scan_id_ls.append(int(spec['scan_id']))

        spectra = rnn.pad_sequence(spectra_ls, batch_first=True).type(model_type)
        precursors = torch.tensor(list(zip(precursor_masses_ls, precursor_charges_ls)), dtype=model_type)
        frac = torch.tensor(frac_ls).type(model_type)
        window_upper = torch.tensor(window_upper_ls, dtype=torch.long)
        window_lower = torch.tensor(window_lower_ls, dtype=torch.long)
        mz1 = torch.tensor(mz1_ls, dtype=torch.float)

        # placeholders (label-dependent)
        peptide_token = None
        fragment_label = None
        theo_mz_bin = None
        missing_mz_bin = None
        peptide_ids = None
        peptide_seq = None
        
        sid = spec.get('scan_id', None)
        if sid is None or (isinstance(sid, float) and np.isnan(sid)):
            scan_id_ls.append(None)
        else:
            scan_id_ls.append(int(sid))
        meta = {"scan_id": scan_id_ls}
        return spectra, precursors, None, frac, None, None, None, None, window_upper, window_lower, mz1, meta
