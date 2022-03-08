import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_heaviside(x, input):
    y = torch.zeros_like(x)
    y[x < 0] = 0
    y[x > 0] = 1
    y[x == 0] = input
    return y

def component_loss_f(prediction, targets):
    #Takes the MSE of each component and returns the array of losses
    loss = torch.zeros(prediction.shape[1])
    L = nn.MSELoss()
    for i in range(prediction.shape[1]):
        loss[i] = L(prediction[:, i], targets[:,i])
    return loss


def component_loss_f_L1(prediction, targets):
    #Takes the MSE of each component and returns the array of losses
    loss = torch.zeros(prediction.shape[1])
    L = nn.L1Loss()
    for i in range(prediction.shape[1]):
        loss[i] = L(prediction[:, i], targets[:,i])
    return loss


def log_loss(prediction, target, nnuc=2):
    # Log Loss Function for standard ML.
    # If there are negative values in X we use MSE
    #Enuc stays with MSE because its normalized

    #X is not allowed to be negative. Enuc is
    X = prediction[:, :nnuc]
    X_target = target[:, :nnuc]
    enuc = prediction[:, nnuc]
    enuc_target = target[:, nnuc]

    L = nn.MSELoss()
    F = nn.L1Loss()

    enuc_loss = L(enuc, enuc_target) + F(torch.sign(enuc), torch.sign(enuc_target))


    #if there are negative numbers we cant use log on mass fractions
    if torch.sum(X < 0) > 0:
        #how much do we hate negative numbers?
        factor = 1000 # a lot
        return enuc_loss + factor*L(X, X_target)

    else:
        barrier = torch.tensor([.1], device=device)
        value = torch.tensor([0.], device=device)

        #greater than barrier we apply mse loss
        #less then barier we apply log of mse loss
        A = my_heaviside(X_target - barrier, value)
        B = -my_heaviside(X_target - barrier, value) + 1

        X_loss =  torch.sum(A * L(X, X_target) + B* torch.abs(.01*L(torch.log(X), torch.log(X_target))))

    return enuc_loss + X_loss

def logX_loss(prediction, target, nnuc=2):
    # We are working with mass fractions in the form of -1/log(X_k)

    X = prediction[:, :nnuc]
    X_target = target[:, :nnuc]
    enuc = prediction[:, nnuc]
    enuc_target = target[:, nnuc]

    L = nn.MSELoss()
    F = nn.L1Loss()

    # enuc is allowed to be negative
    # but penalty should be given if prediction is of different signs
    enuc_fac = 1
    enuc_loss = L(enuc, enuc_target) + enuc_fac * F(torch.sign(enuc), torch.sign(enuc_target))

    # we do not want negative values for mass fractions
    if torch.sum(X < 0) > 0:
        #how much do we hate negative numbers?
        factor = 1000  #a lot
    else:
        factor = 1

    return factor * L(X, X_target) + enuc_loss


def rms_weighted_error(input, target, solution, atol=1e-6, rtol=1e-6):
    error_weight = atol + rtol * torch.abs(solution)
    weighted_error = (input - target) / error_weight
    rms_weighted_error = torch.sqrt((weighted_error**2).sum() / input.data.nelement())
    return rms_weighted_error

def loss_mass_fraction(prediction, nnuc=2):
    return 10* torch.abs(1 - torch.sum(prediction[:, :nnuc]))

def loss_mass_fraction_sum(prediction, totsum=0.5, nnuc=2):
    F = nn.L1Loss()
    total = totsum * torch.ones(prediction.shape[0], device=device)
    
    return F(torch.sum(prediction[:, :nnuc], 1), total)

def loss_mass_fraction_sum_L(prediction, totsum=0.5, nnuc=2):
    L = nn.MSELoss()
    total = totsum * torch.ones(prediction.shape[0], device=device)
    
    return L(torch.sum(prediction[:, :nnuc], 1), total)

def loss_mass_fraction_log(prediction, totsum=1.0, nnuc=2):
    L = nn.MSELoss()
    total = totsum * torch.ones(prediction.shape[0], device=device)
    mass_fraction = torch.exp(-0.5/prediction[:, :nnuc])
    
    return L(torch.sum(mass_fraction, 1), total)

def loss_pure(prediction, target, log_option = False):

    if log_option:
        L = nn.MSELoss()

        #if there are negative numbers we cant use log
        if torch.sum(prediction < 0) > 0:
            #how much do we hate negative numbers?
            factor = 1000 # a lot
            return factor*L(prediction, target[:, :nnuc+1])

        else:
            barrier = torch.tensor([.1], device=device)
            value = torch.tensor([0.], device=device)
            #greater than barrier we apply mse loss
            #less then barier we apply log of mse loss

            A = my_heaviside(target[:, :nnuc+1] - barrier, value)
            B = -my_heaviside(target[:, :nnuc+1] - barrier, value) + 1

            L =  torch.sum(A * L(target[:, :nnuc+1], prediction) + B* torch.abs(.01*L(torch.log(target[:, :nnuc+1]), torch.log(prediction))))

            return L

    else:
        L = nn.MSELoss()
        return L(prediction, target[:, :nnuc+1])


def tanh_loss(dxdt, prediction):
    L = nn.MSELoss()
    out = L(dxdt, prediction)

    return torch.tanh(out)

def signed_loss_function(pred, actual):
    F = nn.L1Loss()
    return F(torch.sign(pred), torch.sign(actual))


def relative_loss(prediction, target):
    threshold = target.clone()
    threshold[target<1.e-15] = 1.e-15

    L = torch.mean(torch.abs(prediction-target)/threshold)

    #If has nan
    if L.isnan().sum().item() > 0:
        return torch.tensor([1.0])
    else:
        return torch.mean(torch.abs(prediction-target)/threshold)
