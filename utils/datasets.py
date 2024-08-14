import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from mnist.loader import MNIST
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    # print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }


class MorphomnistDataset(Dataset):

    def __init__(self, root_dir, transform=None, test=False, gz=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        mndata = MNIST(root_dir, gz=gz)

        if not test:
            images, labels = mndata.load_training()
            self.features = np.genfromtxt(root_dir / 'train-morpho-tas.csv', delimiter=',')[1:, :]
        else:
            images, labels = mndata.load_testing()
            self.features = np.genfromtxt(root_dir / 't10k-morpho-tas.csv', delimiter=',')[1:, :]

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = np.array(self.images[idx]).reshape(28, 28)
        # sample = sample[np.newaxis,:]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.features[idx], self.labels[idx]


class MorphomnistDecodeDataset(Dataset):
    def __init__(self, encodings, features, labels, device='cpu'):
        self.device = device
        self.encodings = torch.from_numpy(encodings).to(device)
        self.features = torch.from_numpy(features).to(device)
        self.labels = torch.from_numpy(labels).to(device)

        self.len = encodings.shape[0]

    # print('data loaded on {}'.format(self.x.device))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.encodings[index], self.features[index], self.labels[index]

    def get_metadata(self):
        return {
            'n': self.len
        }


class UKBBT1Dataset(Dataset):

    def __init__(self, csv_file_path, img_dir, transform=None):
        self.csv_file_path = csv_file_path
        self.img_dir = img_dir

        self.df = pd.read_csv(csv_file_path, low_memory=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(int(self.df.iloc[idx]['eid'])) + '.tiff'
        img = tiff.imread(self.img_dir / img_name)

        if self.transform:
            img = self.transform(img)

        return self.df.iloc[idx]['Sex'], self.df.iloc[idx]['Age'], self.df.iloc[idx]['BMI'], img


class UKBBT13DDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.nii.gz')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        sample = nib.load(img_name).get_fdata()

        if self.transform:
            sample = self.transform(sample)
        return sample


class PRSDataframe(Dataset):

    def __init__(self, data_path):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        #for col in self.df.columns:
        #    print(col)
        #rint(self.df)


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx] # 55:-13 vols, aparc vols, ct
        #return np.hstack((np.array([row['Age'],row['Sex_Numeric'],row['LRRK2_G2019s'],row['APOE4_binary']]), self.df.iloc[idx][50:].values.astype(float))) #-13
        return np.hstack((np.array([row['Age'],  row['Sex_Numeric'], row['APOE4_binary'],row['chr1.154925709.G.C_C_PMVK_rs114138760'],row['chr1.155236246.G.A_A_GBA_T408M_rs75548401'],row['chr1.155236376.C.T_T_GBA_E365K_rs2230288'],row['chr1.226728377.T.C_C_ITPKB_rs4653767'],row['chr1.232528865.C.T_T_SIPA1L2_rs10797576'],row['chr2.101796654.T.C_C_IL1R2.MAP4K4_rs34043159'],row['chr2.134782397.C.T_C_ACMSD.TMEM163_rs6430538'],row['chr2.165277122.C.T_T_SCN3A.SCN2A_rs353116'],row['chr2.168272635.G.T_T_STK39_rs1955337'],row['chr3.18235996.T.G_G_SATB1_rs4073221'],row['chr3.48711556.G.T_G_NCKIPSD.CDC71.IP6K2_rs12497850'],row['chr3.165773492.C.T_T_BuChE_rs1803274'],row['chr3.183044649.G.A_A_MCCC1_rs12637471'],row['chr4.958159.T.C_C_TMEM175_rs34311866'],row['chr4.89704988.G.A_G_SNCA_rs356181'],row['chr4.113439216.T.C_C_ANK2.CAMK2D_rs78738012'],row['chr5.60978096.C.A_C_ELOVL7.NDUFAF2_rs2694528'],row['chr6.32218019.C.T_T_NOTCH4_G1739S_rs8192591'],row['chr6.32698883.C.T_T_HLA_DBQ1_rs115462410'],row['chr7.23254127.A.G_G_GPNMP_rs199347'],row['chr8.11854934.A.C_C_CTSB_rs1293298'],row['chr8.22668467.T.C_T_BIN3_rs2280104'],row['chr9.17579692.T.G_T_SH3GL2_rs13294100'],row['chr10.15527599.C.A_C_FAM171A1.ITGA8_rs10906923'],row['chr10.119950976.C.T_T_MIR4682_rs118117788'],row['chr11.133895472.T.C_T_MIR4697_rs329648'],row['chr12.40220632.C.T_T_LRRK2_rs76904798'],row['chr12.40340400.G.A_A_LRRK2_G2019S_rs34637584'],row['chr12.122819039.A.G_G_OGFOD2.CCDC62_rs11060180'],row['chr14.54882151.C.T_T_GCH1_rs11158026'],row['chr14.88006268.C.T_T_GALC.GPR65_rs8005172'],row['chr16.19268142.T.G_T_COQ7.SYT17_rs11343'],row['chr16.31110472.G.A_A_ZNF646.KAT8.BCKDK_rs14235'],row['chr16.52565276.C.T_T_TOX3.CASC16_rs4784227'],row['chr17.17811787.G.A_A_SREBF1_rs11868035'],row['chr20.3172857.G.A_A_DDRGK1_rs55785911'],row['chr22.19946502.A.G_A_COMT_rs174674'],row['chr22.19950115.T.G_T_COMT_rs5993883'],row['chr22.19957654.A.G_A_COMT_rs740603'],row['chr22.19961340.G.C_G_COMT_rs165656'],row['chr22.19962429.A.G_G_COMT_rs6269'],row['chr22.19962712.C.T_C_COMT_rs4633'],row['chr22.19963684.C.G_G_COMT_rs4818'],row['chr22.19963748.G.A_A_COMT_rs4680'],row['chr22.19969258.G.A_G_COMT_rs165599']]), self.df.iloc[idx][124:].values.astype(float))) # 4:-2 for ukbb data _Numeric 25:-1 5:-1 #row['COHORT'] 5:-1

        # ukbb shortened snps
        #return np.hstack((np.array([row['Age'], row['Sex_Numeric'], row['chr1.155236246.G.A_A_GBA_T408M_rs75548401'] , row['chr1.155236376.C.T_T_GBA_E365K_rs2230288']  , row['chr2.101796654.T.C_C_IL1R2.MAP4K4_rs34043159'] , row['chr2.134782397.C.T_C_ACMSD.TMEM163_rs6430538'] , row['chr2.165277122.C.T_T_SCN3A.SCN2A_rs353116']  , row['chr4.958159.T.C_C_TMEM175_rs34311866'] , row['chr4.76277833.C.T_T_FAM47E.STBD1_rs6812193'] , row['chr4.89761420.A.G_G_SNCA_rs3910105'] , row['chr4.109912954.A.G_G_EGF_rs4444903'] , row['chr4.113439216.T.C_C_ANK2.CAMK2D_rs78738012'] , row['chr5.60978096.C.A_C_ELOVL7.NDUFAF2_rs2694528'] , row['chr6.27713436.G.A_A_ZNF184_rs9468199'] , row['chr6.32218019.C.T_T_NOTCH4_G1739S_rs8192591'] , row['chr8.22668467.T.C_T_BIN3_rs2280104'] , row['chr11.133895472.T.C_T_MIR4697_rs329648'] , row['chr12.40220632.C.T_T_LRRK2_rs76904798'] , row['chr12.40340400.G.A_A_LRRK2_G2019S_rs34637584'] , row['chr14.88006268.C.T_T_GALC.GPR65_rs8005172'] , row['chr15.61701935.G.A_G_VPS13C_rs2414739'] , row['chr16.52565276.C.T_T_TOX3.CASC16_rs4784227'] , row['chr20.3172857.G.A_A_DDRGK1_rs55785911'] , row['chr22.19950115.T.G_T_COMT_rs5993883'] , row['chr22.19963748.G.A_A_COMT_rs4680']]), self.df.iloc[idx][124:].values.astype(float)))
    # , row['chr1.205754444.C.T_C_NUCKS1_rs823118'], row['chr4.950422.A.C_C_TMEM175_rs34884217'] # must have no variability
    #, row['APOE4_Binary'], , row['Sex_Numeric'] # 50 , row['APOE4_binary'], row['Group_numeric'] # chr22.19963748.G.A_A_COMT_rs4680 row['chr4.89704988.G.A_G_SNCA_rs356181']
 # 126
