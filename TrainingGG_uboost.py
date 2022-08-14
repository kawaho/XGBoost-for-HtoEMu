#!/usr/bin/python
from utils import *
import numpy as np
import pandas as pd
import ROOT as r
from sklearn.model_selection import train_test_split
from datetime import timedelta

#Set features to train
#feature_names = ['njets', 'met', 'empt', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']
feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']

#Apply cuts
filters = [('label', '!=', 3), ('e_m_Mass', '>', 100), ('e_m_Mass', '<', 170)]

print("Start loading df")
#Load df
data_vbf = pd.read_parquet('../results/csv4BDT/makeDF_v9_diboson_others_tt_signal_signalAlt_data_oc_gg.parquet', engine='pyarrow', columns=feature_names+['weight','label','e_m_Mass','e_m_Mass_reso'], filters=filters)

print("Finish loading df")
print("Start processing df")

#Use abs weight
data_vbf['absweight'] = data_vbf['weight'].abs()

#De-scale the dilepton pt
data_vbf['empt_Per_e_m_Mass'] = data_vbf['empt']/data_vbf['e_m_Mass']
feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt_Per_e_m_Mass', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']

#Split data into X and y
X = data_vbf
Y = data_vbf['label']

#Split data into train and tsest sets
seed = 123
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#Divide signal by mass resolution
X_train['absweight'].loc[X_train['label']>0] /= X_train[X_train['label']>0]['e_m_Mass_reso']

#Equalize weights
weights_125 = X_train[y_train==1]['absweight'].sum()
for label in [110,120,130,140,150,160]:
  weight = X_train[y_train==label]['absweight'].sum()
  X_train.loc[y_train==label, ['absweight']] *= weights_125/weight

#Train all sig
y_train = (y_train > 0).astype(int)

#Get bkg to sig ratio for reweighting
bkg_to_sig_ratio = get_bkg_to_sig_ratio(X_train, y_train)
X_train[y_train==1]['absweight'] *= bkg_to_sig_ratio

print("Finish processing df")

#Train
feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt_Per_e_m_Mass', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']
print("Start training")
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier
flatnessloss = BinFlatnessLossFunction(uniform_features=['e_m_Mass'], uniform_label=0, n_bins=14)
model = UGradientBoostingClassifier(loss=flatnessloss, train_features=feature_names)
model.fit(X_train, y_train, X_train['absweight'])
print("Finish training")

#Save model
import pickle
filename = '/hadoop/users/kho2/uboost_gg.sav'
pickle.dump(model, open(filename, 'wb'))

