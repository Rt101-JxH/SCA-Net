import numpy as np
import torch


EPS = np.finfo(np.float64).eps


def _object_score(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.double()
    target = target.double()
    foreground_mean = prediction[target == 1].mean()
    foreground_std = prediction[target == 1].std(unbiased=False)
    score = 2.0 * foreground_mean / (foreground_mean**2 + 1.0 + foreground_std + EPS)
    return score.float()


def _s_object(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.double()
    target = target.double()

    foreground_prediction = prediction.clone()
    foreground_prediction[target != 1] = 0.0
    background_prediction = 1.0 - prediction.clone()
    background_prediction[target == 1] = 0.0

    foreground_score = _object_score(foreground_prediction, target)
    background_score = _object_score(background_prediction, 1.0 - target)
    foreground_ratio = target.mean()
    score = foreground_ratio * foreground_score + (1.0 - foreground_ratio) * background_score
    return score.float()


def _centroid(target: torch.Tensor) -> tuple[int, int]:
    if target.sum() == 0:
        return target.shape[0] // 2, target.shape[1] // 2

    coordinates = torch.nonzero(target == 1, as_tuple=False)
    x = coordinates[:, 0].float().mean().round().int()
    y = coordinates[:, 1].float().mean().round().int()
    return x.item(), y.item()


def _divide(target: torch.Tensor, x: int, y: int):
    top_left = target[:x, :y]
    top_right = target[x:, :y]
    bottom_left = target[:x, y:]
    bottom_right = target[x:, y:]

    total = target.numel()
    weights = (
        top_left.numel() / total,
        top_right.numel() / total,
        bottom_left.numel() / total,
        bottom_right.numel() / total,
    )
    return top_left, top_right, bottom_left, bottom_right, *weights


def _ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.double()
    target = target.double()

    prediction_mean = prediction.mean()
    target_mean = target.mean()
    count = prediction.numel()

    prediction_var = ((prediction - prediction_mean) ** 2).sum() / (count - 1 + EPS)
    target_var = ((target - target_mean) ** 2).sum() / (count - 1 + EPS)
    covariance = ((prediction - prediction_mean) * (target - target_mean)).sum() / (count - 1 + EPS)

    alpha = 4.0 * prediction_mean * target_mean * covariance
    beta = (prediction_mean**2 + target_mean**2) * (prediction_var + target_var)

    if alpha.item() != 0:
        score = alpha / (beta + EPS)
    elif beta.item() == 0:
        score = torch.tensor(1.0, device=prediction.device, dtype=torch.float64)
    else:
        score = torch.tensor(0.0, device=prediction.device, dtype=torch.float64)

    return score.float()


def _s_region(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = _centroid(target)
    target_regions = _divide(target, x, y)
    prediction_regions = _divide(prediction, x, y)

    scores = (
        _ssim(prediction_regions[0], target_regions[0]),
        _ssim(prediction_regions[1], target_regions[1]),
        _ssim(prediction_regions[2], target_regions[2]),
        _ssim(prediction_regions[3], target_regions[3]),
    )
    weights = target_regions[4:]
    return scores[0] * weights[0] + scores[1] * weights[1] + scores[2] * weights[2] + scores[3] * weights[3]


def structure_measure(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.double()
    target = target.double()
    foreground_ratio = target.mean()

    if foreground_ratio.item() == 0:
        score = 1.0 - prediction.mean()
    elif foreground_ratio.item() == 1:
        score = prediction.mean()
    else:
        score = 0.5 * _s_object(prediction, target) + 0.5 * _s_region(prediction, target)
        if score.item() < 0:
            score = torch.tensor(0.0, device=prediction.device, dtype=torch.float64)

    return score.float()


def weighted_f_measure(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from scipy.ndimage import convolve, distance_transform_edt

    prediction_np = prediction.detach().cpu().numpy().astype(np.float64)
    target_np = target.detach().cpu().numpy().astype(np.float64)

    error = np.abs(prediction_np - target_np)
    distance, indices = distance_transform_edt(1 - target_np, return_indices=True)

    kernel = np.exp(-((np.mgrid[-3:4, -3:4][0] ** 2 + np.mgrid[-3:4, -3:4][1] ** 2) / 50.0))
    kernel = kernel / kernel.sum()

    transferred = error.copy()
    transferred[target_np != 1] = transferred[indices[0, target_np != 1], indices[1, target_np != 1]]
    smoothed = convolve(transferred, kernel, mode="nearest")
    weighted_error = error.copy()
    weighted_error[(target_np == 1) & (smoothed < error)] = smoothed[(target_np == 1) & (smoothed < error)]

    importance = np.ones_like(target_np)
    importance[target_np != 1] = 2.0 - np.exp(np.log(1.0 - 0.5) / 5.0 * distance[target_np != 1])
    weighted_error = weighted_error * importance

    true_positive = np.sum(target_np) - np.sum(weighted_error[target_np == 1])
    false_positive = np.sum(weighted_error[target_np != 1])
    recall = 1.0 - np.mean(weighted_error[target_np == 1])
    precision = true_positive / (true_positive + false_positive + EPS)
    score = 2.0 * recall * precision / (recall + precision + EPS)
    return torch.tensor(score, device=prediction.device).float()


def calculate_binary_metrics(
    prediction: torch.Tensor, target: torch.Tensor, threshold: float
) -> tuple[float, float, float, float, float, float]:
    threshold = min(threshold, 1.0)
    binary_prediction = (prediction >= threshold).float()

    num_predicted = binary_prediction.sum().item()
    num_background = (binary_prediction == 0).sum().item()
    true_positive = ((binary_prediction == 1) & (target == 1)).float().sum().item()
    num_object = target.sum().item()

    false_negative = num_object - true_positive
    false_positive = num_predicted - true_positive
    true_negative = num_background - false_negative

    if true_positive == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    iou = true_positive / (false_negative + num_predicted)
    precision = true_positive / num_predicted
    recall = true_positive / num_object
    specificity = true_negative / (true_negative + false_positive)
    dice = 2.0 * true_positive / (num_object + num_predicted)
    f_measure = 2.0 * precision * recall / (precision + recall)
    return precision, recall, specificity, dice, f_measure, iou


def _alignment_term(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.double()
    target = target.double()

    prediction_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    return 2.0 * (target_centered * prediction_centered) / (
        target_centered**2 + prediction_centered**2 + EPS
    )


def _enhanced_alignment_term(alignment_matrix: torch.Tensor) -> torch.Tensor:
    return ((alignment_matrix + 1.0) ** 2) / 4.0


def enhanced_measure(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.double()
    target = target.double()

    if target.sum() == 0:
        enhanced = 1.0 - prediction
    elif (1.0 - target).sum() == 0:
        enhanced = prediction.clone()
    else:
        enhanced = _enhanced_alignment_term(_alignment_term(prediction, target))

    score = enhanced.sum() / (target.numel() - 1 + EPS)
    return score.float()
