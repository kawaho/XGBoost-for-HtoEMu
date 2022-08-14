#!/usr/bin/python
from utils import *
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import timedelta
import multiprocessing, tqdm, time, argparse

parser = argparse.ArgumentParser(description='Train BDTs')
parser.add_argument('-c', '--cat', type=str, default='gg', help='category to train (default: %(default)s)')
args = parser.parse_args()

#Get total no of avaliable threads
nthreads = multiprocessing.cpu_count()

#Set features to train
if args.cat=='gg':
  #feature_names = ['njets', 'met', 'j1pt', 'absj1Eta', 'Ht_had', 'j2pt', 'absj2Eta', 'j1_j2_mass', 'DeltaEta_j1_j2', 'DeltaPhi_em_met']
  feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']
  #feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']
else:
  feature_names = ['absmEta', 'abseEta', 'Zeppenfeld_DeltaEta', 'met', 'DeltaEta_e_m', 'empt', 'j1pt', 'DeltaEta_j1_j2', 'j1_j2_mass', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']

#Apply cuts
filters = [('label', '!=', 3), ('e_m_Mass', '>', 100), ('e_m_Mass', '<', 120), ('ept', '>', 25), ('mpt', '>', 20)]
print("Start loading df")
#Load df

#data_vbf = pd.read_parquet(f'../results/csv4BDT/makeDF_v9_diboson_others_tt_signal_signalAlt_data_oc_{args.cat}.parquet', engine='pyarrow', columns=feature_names+['weight','label', 'e_m_Mass', 'sample'], filters=filters)
data_vbf = pd.read_parquet(f'../results/csv4BDT/makeDF_v9_diboson_others_tt_signal_signalAlt_data_oc_{args.cat}.parquet', engine='pyarrow', columns=feature_names+['weight','label', 'e_m_Mass_reso', 'e_m_Mass', 'sample'], filters=filters)
print("Finish loading df")

#Use abs weight
data_vbf = data_vbf[data_vbf['sample'].eq(3)|data_vbf['sample'].eq(10)]
#data_vbf = data_vbf[data_vbf['sample'].eq(3)|data_vbf['sample'].eq(10)|data_vbf['label'].gt(0)]
data_vbf['absweight'] = data_vbf['weight'].abs()

#De-scale the dilepton pt
#data_vbf['empt_Per_e_m_Mass'] = data_vbf['empt']/data_vbf['e_m_Mass']
if args.cat=='gg':
  #feature_names = ['njets', 'met', 'j1pt', 'absj1Eta', 'Ht_had', 'j2pt', 'absj2Eta', 'j1_j2_mass', 'DeltaEta_j1_j2', 'DeltaPhi_em_met']
  feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']
  #feature_names = ['abseEta', 'absmEta', 'njets', 'met', 'empt_Per_e_m_Mass', 'DeltaEta_e_m', 'DeltaPhi_em_met', 'j1pt', 'absj1Eta', 'DeltaEta_j1_em', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']
else:
  feature_names = ['absmEta', 'abseEta', 'Zeppenfeld_DeltaEta', 'met', 'DeltaEta_e_m', 'empt_Per_e_m_Mass', 'j1pt', 'DeltaEta_j1_j2', 'j1_j2_mass', 'Ht_had', 'Rpt', 'pt_cen_Deltapt']

#Split data into X and Y
X = data_vbf
Y = data_vbf['sample']

#Split data into train and test sets
seed = 123
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#Divide signal by mass resolution
X_train['absweight'].loc[X_train['label']>0] /= X_train[X_train['label']>0]['e_m_Mass_reso']

#Equalize weights
#weights_125 = X_train[y_train==1]['absweight'].sum()
#for label in [110,120,130,140,150,160]:
#  weight = X_train[y_train==label]['absweight'].sum()
#  X_train.loc[y_train==label, ['absweight']] *= weights_125/weight

#Train all sig
y_train = (y_train > 3).astype(int)
y_test = (y_test > 3).astype(int)

#Bkg reweight (shape)
#X_train['mass_bin'] = pd.cut(X_train['e_m_Mass'], bins=range(100,171))
#
#groups_sig = X_train[X_train['label']>0].groupby('mass_bin')['absweight'].sum()
#groups_sig /= groups_sig.sum()
#
#groups_bkg = X_train[X_train['label']==0].groupby(['sample', 'mass_bin'])['absweight'].sum()
#groups_bkg /= groups_bkg.groupby(level=0).sum()
#
#for b, bw in groups_bkg.iteritems():
#  if bw!=0: X_train.loc[X_train['sample'].eq(b[0])&X_train['mass_bin'].eq(b[1]), ['absweight']] *= groups_sig[b[1]]/bw  
#
##Plot the reweighting result
#from coffea import hist
#import mplhep
#import matplotlib.pyplot as plt
#plt.style.use(mplhep.style.CMS)
#whichcat = 'VBF' if args.cat=='vbf' else 'ggH'
#emmass = hist.Hist('A.U.', hist.Bin('e_m_Mass', r'$m_{e\mu}$', 70, 100, 170), hist.Cat("sample", f"{whichcat} cat samples"))
#emmass.fill(e_m_Mass=X_train[X_train['label'].gt(0)]['e_m_Mass'].to_numpy(), sample='signal', weight=X_train[X_train['label'].gt(0)]['absweight'].to_numpy())
#emmass.fill(e_m_Mass=X_train[X_train['sample'].eq(3)]['e_m_Mass'].to_numpy(), sample=r'$t\bar{t}$', weight=X_train[X_train['sample'].eq(3)]['absweight'].to_numpy())
#emmass.fill(e_m_Mass=X_train[X_train['sample'].eq(10)]['e_m_Mass'].to_numpy(), sample='WW', weight=X_train[X_train['sample'].eq(10)]['absweight'].to_numpy())
#fig, ax = plt.subplots(figsize=(12,12))
#hist.plot1d(emmass, density=True)
#lumi = mplhep.cms.label(ax=ax, lumi=138, label="Preliminary", data=1)
##ax.legend(title_fontsize=16, title='VBF cat', fontsize=16)
#fig.savefig(f'plots/BDTValid/bkg_{args.cat}_reweight.png',bbox_inches='tight')

#Equalize bkg
#X_train = X_train.assign(
#         sigtotal =   np.where(X_train['label']  > 0,   X_train.absweight, 0),
#         bkgtotal3 =  np.where(X_train['sample'] == 3,  X_train.absweight, 0),
#         bkgtotal10 = np.where(X_train['sample'] == 10, X_train.absweight, 0)
#      )
#groups = X_train.groupby('mass_bin')
#lut = groups.agg({'sigtotal':sum, 'bkgtotal3':sum, 'bkgtotal10':sum})
#lut_3 = (lut['sigtotal']/lut['bkgtotal3']).replace(np.nan, 0)
#lut_10 = (lut['sigtotal']/lut['bkgtotal10']).replace(np.nan, 0)
#
#tic = time.time()
#for emmass in range(100,170,1):
#  print(emmass)
#  X_train.loc[X_train['sample'].eq(3) &  X_train.e_m_Mass.ge(emmass) & X_train.e_m_Mass.lt(emmass+1), ['absweight']] *= lut_3.loc[emmass+0.5]
#  X_train.loc[X_train['sample'].eq(10) & X_train.e_m_Mass.ge(emmass) & X_train.e_m_Mass.lt(emmass+1), ['absweight']] *= lut_10.loc[emmass+0.5]
#toc = time.time()
#print("Time spent on reweighting method 2 %s"%str(timedelta(seconds=toc - tic)))

#Get bkg to sig ratio for reweighting
bkg_to_sig_ratio = get_bkg_to_sig_ratio(X_train, y_train)
print("Finish processing df")

#Train
print("Start training")
# Train with the default model
tic = time.time()
if args.cat=='gg':
  model = xgb.XGBClassifier(n_jobs=nthreads, objective='binary:logistic', scale_pos_weight=bkg_to_sig_ratio, max_depth=3, use_label_encoder =False)#, reg_lambda=10, reg_alpha=10, seed=123, learning_rate=0.1, n_estimators=50)
else:
  model = xgb.XGBClassifier(n_jobs=nthreads, objective='binary:logistic', scale_pos_weight=bkg_to_sig_ratio, max_depth=3, use_label_encoder =False, reg_lambda=10, reg_alpha=10)#, seed=123, learning_rate=0.1, n_estimators=50)

model.fit(X_train[feature_names], y_train, early_stopping_rounds=10, sample_weight=X_train['absweight'], eval_set = [(X_train[feature_names], y_train), (X_test[feature_names], y_test)], eval_metric='auc', sample_weight_eval_set = [X_train['absweight'], X_test['absweight']], verbose=True, callbacks=[xgb_progress(100)])
toc = time.time()
print("Finish training")
print("Time spent on training %s"%str(timedelta(seconds=toc - tic)))

#Save model
model_out_file = f"results/model_{args.cat}_v9_bkgbkg.json"
model.save_model(model_out_file)
