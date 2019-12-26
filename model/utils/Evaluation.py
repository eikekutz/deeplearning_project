from skimage.measure import label, regionprops
from sklearn.metrics import f1_score

def F1_score(Y_val, Y_pred):

  Y_val_label=Y_val.numpy()
  Y_pred_label=Y_pred.numpy()

  F1_score=f1_score(Y_val.reshape(-1), Y_pred.reshape(-1), average='macro')


  return F1_score