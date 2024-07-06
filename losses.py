import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def sdwp(logits, label): 
    logits_ =  logits.softmax(1)
    eye = torch.eye(logits_.shape[1]).cuda()
    logits_y = (logits_ * eye[label]).sum(1)
    logits_y = logits_y.view(-1, 1)    
    diff =   logits_ -  logits_y  #torch.mean(logits_, dim=1)               
    var  = (torch.sum(torch.square(diff), dim=1)) /logits_.shape[1] - 1
    std =  torch.sqrt(var)
    std_ =  std  
    return  std_.detach() 


def margin(logit, target):
    eye = torch.eye(10).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest = True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    return  probs_GT - probs_2nd 



def adapt_pert(logit, target,ep, weight= 0.52, weight_type = "margin"):
    mg_ = margin(logit, target)
    sd_ = sdwp(logits, target)
    if weight_type == "margin":
      wei = torch.exp(weight * mg_).detach() 
    else:
      wei = torch.exp(weight * sd_).detach() 
    eps =  wei * ep
    eps = torch.unsqueeze(eps, dim = 0) 
    epsl = eps.view(-1, 1, 1, 1)
    epsl =  epsl.cuda()
    step = epsl / 4
    return epsl, step



def at_loss_initial(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
             
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + (step_size/2)   * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - (epsilon/2)), x_natural + (epsilon/2))
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
   
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()    ##Changr to train

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    
 
    loss_natural = F.cross_entropy(logits, y)
    loss_adv = F.cross_entropy(model(x_adv), y)  
    
    return loss_adv




def at_adapt(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=5.0,
              distance='l_inf'):
   
    model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    
    
       
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    epsl, step =  adapt_pert(model(x_natural),y,epsilon)  
  
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()   
           
            with torch.enable_grad():
                loss_ce =  F.cross_entropy(model(x_adv), y) 
               
     
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]

            x_adv = x_adv.detach() + step * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsl), x_natural + epsl) 
            x_adv = torch.clamp(x_adv, 0.0, 1.0)      
        
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
      
    model.train() 
   
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    loss_nat  = F.cross_entropy(logits, y)
    logits_adv = model(x_adv)

    loss_adv = F.cross_entropy(logits_adv, y) 

    return   loss_adv
