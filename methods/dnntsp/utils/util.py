import json
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import sys
sys.path.append("..")


# convert data from cpu to gpu, accelerate the running speed
def convert_to_gpu(data):
    if torch.cuda.is_available():
        data = data.to(torch.device('cuda'))
    return data


def convert_all_data_to_gpu(*data):
    res = []
    for item in data:
        item = convert_to_gpu(item)
        res.append(item)
    return tuple(res)


def convert_train_truth_to_gpu(train_data, truth_data):
    train_data = [[convert_to_gpu(basket) for basket in baskets] for baskets in train_data]
    truth_data = convert_to_gpu(truth_data)
    return train_data, truth_data




def get_truth_data(truth_data, total_items):
    """
    Args:
        truth_data: list, shape (baskets_num, items_num)
    Returns:
        turth: tensor, shape (baskets_num, items_total)
    """
    truth_list = []
    for basket in truth_data:
        one_hot_items = F.one_hot(basket, num_classes=total_items)
        one_hot_basket, _ = torch.max(one_hot_items, dim=0)
        truth_list.append(one_hot_basket)
    truth = torch.stack(truth_list)

    return truth


# padding for all users
def pad_sequence(data_list):
    """
    :param data_list: shape (batch_users, baskets, item_embed_dim)
    :return:
    """
    length = [len(sq) for sq in data_list]
    # pad in time dimension
    data = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
    return data, length

