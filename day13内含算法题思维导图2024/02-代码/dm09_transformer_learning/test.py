# # import torch
# # input_ids = torch.tensor([[1, 2, 3, 4]])
# # rand = torch.rand(input_ids.shape)
# # print(rand)
# # mask_arr = (rand < 0.15) * (input_ids != 4)
# # print(mask_arr)
# # input_ids[mask_arr] = 6
# # print(input_ids)
# import random
# print(random.randint(0, 3))
from transformers import BertModel
bert_model = BertModel.from_pretrained('../dm08_transformers/bert-base-chinese')
print(bert_model)