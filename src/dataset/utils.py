import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple, TypeVar
import cv2
import numpy as np
from tqdm import tqdm

#for label propagation
import faiss
import torch.nn.functional as F
import scipy
import scipy.stats
from faiss import normalize_L2
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

A = TypeVar("A")
B = TypeVar("B")


def is_image_file(filename: str) -> bool:
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_root: str,
                 data_list: str,
                 class_list: List[int],
                 remove_images_with_undesired_classes: bool = False,
                 keep_small_area_classes: bool = False) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    """
        Recovers all tuples (img_path, label_path) relevant to the current experiments (class_list
        is used as filter)

        input:
            data_root : Path to the data directory
            data_list : Path to the .txt file that contain the train/test split of images
            class_list: List of classes to keep
        returns:
            image_label_list: List of (img_path, label_path) that contain at least 1 object of a class
                              in class_list
            class_file_dict: Dict of all (img_path, label_path that contain at least 1 object of a class
                              in class_list, grouped by classes.
    """
    image_label_list: List[Tuple[str, str]] = []
    list_read = open(data_list).readlines()

    print(f"Processing data for {class_list}")
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    process_partial = partial(process_image, data_root=data_root, class_list=class_list,
                              remove_images_with_undesired_classes=remove_images_with_undesired_classes,
                              keep_small_area_classes=keep_small_area_classes)

    with Pool(os.cpu_count() // 2) as pool:
        for sublist, subdict in pool.map(process_partial, tqdm(list_read)):  # mmap
            image_label_list += sublist
            for (k, v) in subdict.items():
                class_file_dict[k] += v
        pool.close()
        pool.join()

    return image_label_list, class_file_dict


def process_image(line: str,
                  data_root: str,
                  class_list: List,
                  remove_images_with_undesired_classes: bool,
                  keep_small_area_classes: bool) -> Tuple[List, Dict]:
    """
        Reads and parses a line corresponding to 1 file

        input:
            line : A line corresponding to 1 file, in the format path_to_image.jpg path_to_image.png
            data_root : Path to the data directory
            class_list: List of classes to keep

    """
    line = line.strip()
    line_split = line.split(' ')
    image_name = os.path.join(data_root, line_split[0])
    label_name = os.path.join(data_root, line_split[1])
    item: Tuple[str, str] = (image_name, label_name)
    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    label_class = np.unique(label).tolist()

    if 0 in label_class:
        label_class.remove(0)
    if 255 in label_class:
        label_class.remove(255)
    for label_class_ in label_class:
        assert label_class_ in list(range(1, 81)), label_class_

    c: int
    new_label_class = []
    for c in label_class:
        if c in class_list:
            tmp_label = np.zeros_like(label)
            target_pix = np.where(label == c)
            tmp_label[target_pix[0], target_pix[1]] = 1
            if tmp_label.sum() >= 16 * 32 * 32 or keep_small_area_classes:
                new_label_class.append(c)
        elif remove_images_with_undesired_classes:
            new_label_class = []
            break

    label_class = new_label_class

    image_label_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    if len(label_class) > 0:
        image_label_list.append(item)

        for c in label_class:
            assert c in class_list
            class_file_dict[c].append(item)

    return image_label_list, class_file_dict


def label_propagation(pred_prob, feat, label, args, log, alpha=0.99, max_iter=20, ret_acc=False):
    """
    Args: 
        pred_label: current predicted label [B*C]
        feat: feature embedding for all samples (used for computing similarity) [B*D]
        label: GT label

        alpha:
        max_iter:
    """
    pred_label = pred_prob if args.lp_type > 0 else np.argmax(pred_prob, axis=1)

    # kNN search for the graph
    N, d = feat.shape[0], feat.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index  index = faiss.IndexFlatL2(d)

    normalize_L2(feat)
    index.add(feat)
    # log('n total {}'.format(index.ntotal))
    D, I = index.search(feat, args.k + 1)

    # Create the graph
    if args.w_type == 'poly':
        D = D[:, 1:] ** 3  # [N, k]
    else:
        D = np.exp( (D[:, 1:]-1) / args.gamma )
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (args.k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, args.num_classes_tr+args.num_classes_val))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(args.num_classes_tr+args.num_classes_val):
        if args.lp_type == 0:
            y = np.zeros((N,))
            cur_idx = np.where(pred_label==i)[0]   # pred_label [N]
            y[cur_idx] = 1.0 / (cur_idx.shape[0] + 1e-10)
        else:
            y = pred_label[:, i] / np.sum(pred_label[:, i])
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)   # Use Conjugate Gradient iteration to solve Ax = b
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), p=1, dim=1).numpy()
    probs_l1[probs_l1 < 0] = 0

    new_pred = np.argmax(probs_l1, 1)
    new_acc = float(np.sum(new_pred == label)) / len(label)
    # mean_acc, _ = compute_acc(label, new_pred)
    log('After label propagation Acc: {:.2f}%, Mean Acc: {:.2f}%'.format(new_acc*100, mean_acc*100))
    
    if ret_acc:
        return new_pred, probs_l1, mean_acc, new_acc
    else: 
        return new_pred, probs_l1
