import torch

def mse(y, y_pred) -> torch.Tensor:
    return torch.mean((y-y_pred)**2)

def boundary_loss(prediction, t_boundary) -> torch.Tensor:
    ''' The **boundary loss**, tries to ensure that the solution learned by the PINN matches the initial conditions of the system.


    Returns the loss function for 2 initial conditions
    '''

    # minimising first condition that x(t=0) = 1 (see data notebook)
    loss1 = (torch.squeeze(prediction) - 1)**2

    dxdt = torch.autograd.grad(prediction, t_boundary, torch.ones_like(prediction), create_graph=True)[0]
    loss2 = (torch.squeeze(dxdt) - 0)**2

    return loss1, loss2

def physics_loss(prediction, t_physics, mu, k) -> torch.Tensor:
    ''' The **physics loss**, tries to ensure that the PINN solution obeys the underlying differential equation (see data notebook).


    
    Return the loss function for the harmonic oscillator DE 
    '''


    dxdt = torch.autograd.grad(prediction, t_physics, torch.ones_like(prediction), create_graph=True)[0]

    d2xdt2 = torch.autograd.grad(dxdt, t_physics, torch.ones_like(dxdt), create_graph=True)[0]
    loss = torch.mean((d2xdt2 + mu*dxdt + k*prediction)**2)

    return loss


def physics_loss_imp(prediction, t_physics, mu, k) -> torch.Tensor:
    # compute the "physics loss"

    dx  = torch.autograd.grad(prediction, t_physics, torch.ones_like(prediction), create_graph=True)[0]# computes dy/dx
    dx2 = torch.autograd.grad(dx, prediction, torch.ones_like(dx), create_graph=True)[0]# computes d^2y/dx^2
    physics = dx2 + mu*dx + k*prediction# computes the residual of the 1D harmonic oscillator differential equation
    loss = torch.mean(physics**2)

    return loss