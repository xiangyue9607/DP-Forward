import math
import numpy as np
# from transformers import BertForMaskedLM
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
from prv_accountant import Accountant


# Matrix Gaussian Noise
def matrix_gaussian_noise(epsilon, delta, sensitivity):
    def function_phi(t):
        return (1 + math.erf(t / math.sqrt(2))) / 2

    def B_plus_function(v, epsilon):
        return function_phi(math.sqrt(epsilon * v)) - math.exp(epsilon) * function_phi(-math.sqrt(epsilon * (v + 2)))

    def B_minus_function(u, epsilon):
        return function_phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * function_phi(-math.sqrt(epsilon * (u + 2)))

    def compute_R(epsilon, delta, iterations=5000):
        delta_0 = function_phi(0) - math.exp(epsilon) * function_phi(-math.sqrt(2 * epsilon))
        # print(delta_0)

        start, end = 0, 1e5

        B_function = B_plus_function if delta >= delta_0 else B_minus_function
        for i in range(iterations):
            mid = (start + end) / 2
            value = B_function(mid, epsilon)
            if value < delta:
                end = mid
            else:
                start = mid

        u_star = end
        b_value = B_function(end, epsilon)

        if delta >= delta_0:
            alpha = math.sqrt(1 + u_star / 2) - math.sqrt(u_star / 2)
        else:
            alpha = math.sqrt(1 + u_star / 2) + math.sqrt(u_star / 2)

        R = math.sqrt(2 * epsilon) / alpha

        return R

    R = compute_R(epsilon, delta)
    noise_b = sensitivity / R
    # noise_matrix = noise_b * np.random.normal(size=dim)

    return noise_b


# https://en.m.wikipedia.org/wiki/Additive_noise_mechanisms#Laplace_Mechanism
def Lap_noise(epsilon, dim, sensitivity):
    noise_b = sensitivity / epsilon
    noise_matrix = np.random.laplace(loc=0, scale=noise_b, size=dim)
    return noise_matrix


# https://en.m.wikipedia.org/wiki/Additive_noise_mechanisms#Gaussian_Mechanism
def Gaussian_noise(epsilon, delta, sensitivity):
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_b = sensitivity * sigma
    # noise_matrix = np.random.normal(loc=0,scale=noise_b,size=dim)
    return noise_b


# MVG@CCS18
def MVG_noise(epsilon, delta, dim, sensitivity, gamma):
    def cal_harmonic_number(r, m=1.0):
        sum = 0
        for i in range(1, r + 1):
            sum += 1 / i ** m
        return sum

    r = min(dim)
    H_r = cal_harmonic_number(r)
    H_r_half = cal_harmonic_number(r, m=0.5)
    alpha = (H_r + H_r_half) * gamma ** 2 + 2 * H_r * gamma * sensitivity
    zeta = 2 * math.sqrt(-dim[0] * dim[1] * math.log(delta)) - 2 * math.log(delta) + dim[0] * dim[1]
    beta = 2 * ((dim[0] * dim[1]) ** 0.25) * H_r * sensitivity * zeta
    noise_b = (2 * alpha * ((dim[0] * dim[1]) ** 0.25)) / (-beta + math.sqrt(beta ** 2 + 8 * alpha * epsilon))
    #
    # precisionBudget = ((-beta + np.sqrt(beta ** 2 + 8 * alpha * epsilon)) ** 2) / (4 * (alpha ** 2))
    return noise_b


def gamma_simulation(num_simulation=100000):
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    total_num = embedding_matrix.shape[0]
    total_indices = np.arange(total_num)
    norms = []
    for i in tqdm(range(num_simulation)):
        np.random.shuffle(total_indices)
        sampled_indices = total_indices[:512]
        embeddings = embedding_matrix[sampled_indices]
        norm2 = np.linalg.norm(embeddings)
        norms.append(norm2)
    print(max(norms))
    return max(norms)


def cal_sensitivity_bert_embedding():
    """
    This function may run up to a few minutes. The results are:
    L1 Sensitivity: 56.33
    L2 Sensitivity: 2.89
    """

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # l1-sensitivity
    distance = cdist(embedding_matrix, embedding_matrix, "cityblock")
    l1_sensitivity = np.max(distance)
    print(l1_sensitivity)

    # l2-sensitivity
    distance = cdist(embedding_matrix, embedding_matrix, "euclidean")
    l2_sensitivity = np.max(distance)
    print(l2_sensitivity)

    return l1_sensitivity, l2_sensitivity


def get_noise_multiplier(eps, delta, batch_size=1, dataset_size=50000, epoch=3, local_dp=False, noise_type='aGM'):
    if local_dp:
        if noise_type == 'aGM':
            return matrix_gaussian_noise(epsilon=eps,delta=delta,sensitivity=1)
        elif noise_type == 'GM':
            return Gaussian_noise(epsilon=eps,delta=delta,sensitivity=1)

    start_noise_multiplier = 0.2
    end_noise_multiplier = 50
    while True:
        mid_noise_multiplier = (start_noise_multiplier + end_noise_multiplier) / 2
        accountant = Accountant(
            noise_multiplier=mid_noise_multiplier,
            sampling_probability=batch_size / dataset_size,
            delta=1e-5,
            eps_error=0.1,
            max_compositions=round(dataset_size / batch_size) * epoch,
        )
        _, eps_estimate, _ = accountant.compute_epsilon(
            num_compositions=round(dataset_size / batch_size) * epoch)
        # print(eps_estimate)
        if abs(eps_estimate - eps) < 0.0001:
            break
        # less noise
        if eps_estimate > eps:
            start_noise_multiplier = mid_noise_multiplier
        else:
            end_noise_multiplier = mid_noise_multiplier

    return mid_noise_multiplier



def add_noise(embeddings,noise_factor=0.5,norm_c=1.0,add_noise=True):
    if add_noise:
        embeddings = _max_norm_clip(embeddings, norm_c)
        noise_embeds = noise_factor * torch.normal(mean=0, std=1, size=embeddings.shape).to(embeddings.device)
        embeddings = noise_embeds + embeddings
    return embeddings


def _max_norm_clip(embeddings, norm_c=1.0):
    shape = embeddings.shape
    embeddings = embeddings.reshape(shape[0], -1)
    total_norm = torch.norm(embeddings, dim=-1)
    # print(total_norm.mean())
    clip_coef = norm_c / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    embeddings = torch.multiply(embeddings, clip_coef_clamped.unsqueeze(-1))
    embeddings = embeddings.reshape(shape)

    return embeddings
