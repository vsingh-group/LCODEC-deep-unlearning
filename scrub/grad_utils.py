import torch

def getGradObjs(model):
    grad_objs = {}
    param_objs = {}
    for module in model.modules():
    #for module in model.modules():
        for (name, param)  in module.named_parameters():
            grad_objs[(str(module), name)] = param.grad
            param_objs[(str(module), name)] = param.data

    return grad_objs, param_objs

def gradNorm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1. / 2)
    return total_norm

def getHessian(dw1, dw2=None, approxType='FD', w1=None, w2=None, hessian_device='cpu'):
    original_device = dw1.device
    dw1 = dw1.to(hessian_device)
    dw2 = dw2.to(hessian_device)
    if approxType == 'FD':
        w1 = w1.to(hessian_device)
        w2 = w2.to(hessian_device)
    
        #hessian = torch.matmul((dw1 - dw2), (dw1 - dw2).transpose())
        grad_diff_outer = torch.einsum('p,q->pq', (dw1-dw2), (dw1-dw2))
 
        # divide by weight diff:
        pdist = torch.nn.PairwiseDistance(p=1)
        weight_scaling = pdist(w1.view(1,-1), w2.view(1,-1))

        hessian = torch.div(grad_diff_outer, weight_scaling)

    elif approxType == 'Fisher':
        hessian = torch.einsum('p,q->pq', dw1, dw1)

    else:
        error('Unknown Hessian Approximation Type')

    return hessian.to(original_device)


def getOldPandG(outString, epoch, model, slices_to_update, device):

    name = outString + '_epoch_'  + str(epoch) + "_params.pt"
    paramlist = torch.load(name)
    name = outString + '_epoch_'  + str(epoch) + "_grads.pt"
    gradlist = torch.load(name)
    vectG, vectP, _ = getVectorizedGrad(gradlist, model, slices_to_update, device, paramlist=paramlist)

    return vectG, vectP

 
def getVectorizedGrad(gradlist, model, slices_to_update, device, paramlist=None):

    mapDict = {}
    vect_grad = torch.Tensor(0).to(device)
    vect_param = torch.Tensor(0).to(device)

    for layerIdx in range(len(slices_to_update)):

        [layer, sliceID] = slices_to_update[layerIdx]

        for param in layer.named_parameters():

            orig_shape = param[1][sliceID].shape

            if paramlist is not None:
                pparam = paramlist[(str(layer), param[0])]
                vectVersionParam = torch.flatten(pparam[sliceID])
            else:
                vectVersionParam = torch.flatten(param[1][sliceID])

            pgrad = gradlist[(str(layer), param[0])]
            vectVersionGrad = torch.flatten(pgrad[sliceID])

            start_idx = vect_grad.shape[0]
            vect_grad = torch.cat([vect_grad, vectVersionGrad], dim=0)
            vect_param = torch.cat([vect_param, vectVersionParam], dim=0)
            end_idx = vect_grad.shape[0]

            myKey = (str(layer), param[0], sliceID)
            myVal = [start_idx, end_idx, orig_shape, param[1]]
            mapDict[myKey] = myVal

    return vect_grad, vect_param, mapDict


