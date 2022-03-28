import torch


def detection_rate(output, target, thresh=0.5):
    output = output.view(-1)
    target = target.view(-1)
    correct = 0
    n_one = 0
    with torch.no_grad():
        pred = output >= thresh
        assert pred.shape[0] == len(target)
        correct += torch.sum((pred==1)*(target==1)).item()
        n_one += torch.sum(target==1).item()
    return correct / (1 if n_one == 0 else n_one)


def false_alarm_rate(output, target, thresh=0.5):
    output = output.view(-1)
    target = target.view(-1)
    correct = 0
    n_zero = 0
    with torch.no_grad():
        pred = output >= thresh
        assert pred.shape[0] == len(target)
        correct += torch.sum((pred==1)*(target==0)).item()
        n_zero += torch.sum(target==0).item()
    return correct / (1 if n_zero == 0 else n_zero)


def detection_atper(output, target, target_far=0.01, eps=1e-5, max_iter=100):
    min_thresh = 0
    max_thresh = 1
    thresh = 0.5
    for i in range(max_iter):
        far = false_alarm_rate(output, target, thresh=thresh)
        if far == target_far:
            break
        elif far < target_far:
            # reduce threshold
            max_thresh = thresh
            thresh = 0.5 * (max_thresh + min_thresh)
        elif far > target_far:
            # increase threshold
            min_thresh = thresh
            thresh = 0.5 * (max_thresh + min_thresh)
        if abs(far - target_far) < eps:
            break
    return detection_rate(output, target, thresh=thresh)


def mean_sq_err(output, target):
    with torch.no_grad():
        return torch.mean((output.view(-1) - target.view(-1))**2)



def mine_estimate(T_joint, T_marginal):
    with torch.no_grad():
        return (torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))).item()
