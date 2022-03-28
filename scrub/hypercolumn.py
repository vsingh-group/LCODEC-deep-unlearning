import torch
import torch.nn as nn
import torch.nn.functional as F

class NLP_ActivationsHook(nn.Module):

    def __init__(self, model):
        super(NLP_ActivationsHook, self).__init__()
        self.model = model
        self.model.eval()

        self.layers = []
        self.activations = []
        self.hooks = []

        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(self.nlplinearhook))
                self.layers.append(m)

    def getLayers(self):
        return self.layers
                
    def nlplinearhook(self, module, input, output):
        if len(output.shape) == 3:
            flat = output.mean(dim=[0,1]) # mean over tokens? ; output is of shape 1 *n_tokens * 768 (hidden_emb_size), also mean over batches_dim_0
            self.activations.append(flat)
        else:
            assert len(output.shape) == 2
            output = output.mean(dim=[0])   # mean over batches_dim_0
            self.activations.append(output) # for last two linear layers which only utilize the [CLS] token
        

    def get_NLP_Activations(self, x):
        self.activations = []
        output = self.model(**x)
        return self.activations, output

    def clearHooks(self):
        for x in self.hooks:
            x.remove()

    def __del__(self):
        self.clearHooks()


class ActivationsHook(nn.Module):

    def __init__(self, model):
        super(ActivationsHook, self).__init__()
        self.model = model
        self.model.eval()

        self.layers = []
        self.activations = []
        self.hooks = []

        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(self.hook))
                self.layers.append(m)
            elif isinstance(m, nn.Conv2d):
                self.hooks.append(m.register_forward_hook(self.convhook))
                self.layers.append(m)

    def getLayers(self):
        return self.layers
                
    def hook(self, module, input, output):

        # for batch size > 1
        output = output.mean(dim=[0])
        self.activations.append(output)

    def convhook(self, module, input, output):

        # for batch size > 1, and pixels (choose filters)
        flat = output.mean(dim=[0,2,3]) 
        self.activations.append(flat)

    def getActivations(self, x):
        self.activations = []
        output = self.model(x)
        return self.activations, output

    def clearHooks(self):
        for x in self.hooks:
            x.remove()

    def __del__(self):
        self.clearHooks()

# old class for using hypercolumn to get samples, pixels in activation map are samples for that kernel/filter. Deprecated
class HyperC(nn.Module):
    def __init__(self, model, interp_size=64, interpolate=False):
        super(HyperC, self).__init__()
        self.interpolate = interpolate
        self.interp_size = interp_size
        self.model = model
        self.model.eval()

        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(self.hook)
                self.layers.append(m)
                
    def hook(self, module, input, output):
        if self.interpolate:
            output = F.interpolate(output, size=self.interp_size, mode='bilinear', align_corners=True)
        self.outputs.append(output)


    def getHC(self, x):
        self.outputs = []
        
        _ = self.model(x)
        
        if self.interpolate:
            outputs = torch.cat(self.outputs, dim=1)
        else:
            outputs = self.outputs

        return outputs, self.layers

