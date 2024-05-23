import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.autograd import Variable


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for the original GAN.
    
    Use the torch.nn.functional.binary_cross_entropy_with_logits rather than softmax followed by BCELoss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # True labels are 1 for real data
    true_labels = Variable(torch.ones_like(logits_real))
    # True labels are 0 for fake data
    fake_labels = Variable(torch.zeros_like(logits_fake))
    
    # Calculate the loss for real and fake data separately
    real_loss = bce_loss(logits_real, true_labels)
    fake_loss = bce_loss(logits_fake, fake_labels)
    
    loss = (real_loss + fake_loss) / 2
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss for the original GAN.
    
    Use the torch.nn.functional.binary_cross_entropy_with_logits rather than softmax followed by BCELoss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
#     loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # True labels are 0 for fake data
    true_labels = torch.ones_like(logits_fake)
    
    # Calculate the loss:
    loss = bce_loss(logits_fake, true_labels)
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the LSGAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """    
#     loss = None    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # LSGAN wants to push scores of real data towards 1 and fake data towards 0
    real_loss = torch.mean((scores_real - 1) ** 2)
    fake_loss = torch.mean(scores_fake ** 2)
    
    # Average the losses
    loss = (real_loss + fake_loss) / 2
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the LSGAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """    
#     loss = None    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # LSGAN wants to push scores of fake data towards 1
    loss = torch.mean((scores_fake - 1) ** 2)
    ##########       END      ##########    
    return loss