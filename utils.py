import tqdm 
import xgboost as xgb
def get_bkg_to_sig_ratio(X_train, y_train):
  #Rescale signal weight to bkg weight
  total_bkg_train_weights = X_train.loc[y_train == 0]['absweight'].sum()
  total_sig_train_weights = X_train.loc[y_train == 1]['absweight'].sum()
  bkg_to_sig_ratio = total_bkg_train_weights/total_sig_train_weights
  print('Total Bkg Training weights', 'Total Sig Training weights')
  print(total_bkg_train_weights, total_sig_train_weights)
  print(bkg_to_sig_ratio)
  return bkg_to_sig_ratio

class xgb_progress(xgb.callback.TrainingCallback):
  def __init__(self, rounds):
    self.pbar = tqdm.tqdm(total=rounds)

  def after_iteration(self, model, epoch, evals_log):
    self.pbar.update(1)
