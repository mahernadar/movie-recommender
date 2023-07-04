
import torch
import torch.nn as nn

import torch.nn.functional as F
from utils import AGES_EMBEDDINGS, GENDERS_EMBEDDINGS, MOVIES_EMBEDDINGS


# copied from fastai:
def trunc_normal_(x, mean=0.0, std=1.0):
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

class FMModel(nn.Module):
    def __init__(self, n, k):
        super().__init__()

        self.w0 = nn.Parameter(torch.zeros(1))
        self.bias = nn.Embedding(n, 1)
        self.embeddings = nn.Embedding(n, k)

        # See https://arxiv.org/abs/1711.09160
        with torch.no_grad():
            trunc_normal_(self.embeddings.weight, std=0.01)
        with torch.no_grad():
            trunc_normal_(self.bias.weight, std=0.01)

    def forward(self, X):
        emb = self.embeddings(X)
        # calculate the interactions in complexity of O(nk) see lemma 3.1 from paper
        pow_of_sum = emb.sum(dim=1).pow(2)
        sum_of_pow = emb.pow(2).sum(dim=1)
        pairwise = (pow_of_sum - sum_of_pow).sum(1) * 0.5
        bias = self.bias(X).squeeze().sum(1)
        return torch.sigmoid(self.w0 + bias + pairwise) * 5.5


# def movie_recommender(query:dict, model_path: Union[str, Path], k=10) -> list:
#     """recommender based on nmf model

#     Args:
#         query (dict): _description_
#         model (_type_): _description_
#         k (int, optional): _description_. Defaults to 10.

#     Returns:
#         list: topk movies
#     """
#     return NotImplemented

# model = torch.load("movie-recommender/models/recommender_1_m.pt")
# print(model)


if __name__ == "__main__":
    # top3 =  random_recommender()
    # print(top3)
    model = model = torch.load("movie-recommender/models/recommender_1_m.pt")
    print(model)