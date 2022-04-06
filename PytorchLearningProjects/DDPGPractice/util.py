
import os
import torch
from torch.autograd import Variable


def prRed(prt): print("91m {}" .format(prt))
def prGreen(prt): print("92m {}" .format(prt))
def prYellow(prt): print("93m {}" .format(prt))
def prLightPurple(prt): print("[94m {}" .format(prt))
def prPurple(prt): print("95m {}" .format(prt))
def prCyan(prt): print("96m {}" .format(prt))
def prLightGray(prt): print("97m {}" .format(prt))
def prBlack(prt): print("98m {}" .format(prt))

def to_numpy(var):
    return var.data.numpy()

def to_tensor(numpy_ndarray, volatile=False, requires_grad=False, dtype=torch.FloatTensor):    # 这里传进来的是一个observation
    # print("numpy data : ", numpy_ndarray)
    return Variable(torch.from_numpy(numpy_ndarray), volatile = volatile, requires_grad=requires_grad).type(dtype)
    
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir






