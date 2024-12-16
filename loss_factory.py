import torch
import torch.nn as nn



def cosine_sim(s, im):
    """Cosine similarity between all the image and sentence pairs
    """
    return s.mm(im.t())

class Loss(nn.Module):

    def __init__(self, parser):
        super(Loss, self).__init__()
        self.parser = parser
        self.itc_loss = ContrastiveLoss(margin=parser.margin)
        self.itm_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.itc_value = 0
        self.itm_value = 0
   
    def forward(self, image_embeddings, text_embeddings, logits, labels):
        """
        image_embeddings: (2*batch_size, embedding_dim)
        text_embeddings: (2*batch_size, embedding_dim)
        logits: (2*batch_size,)
        labels: (2*batch_size,)
        """
        positive_mask = labels == 1
        self.itc_value = self.itc_loss(image_embeddings[positive_mask], text_embeddings[positive_mask])
        self.itm_value = self.itm_loss(logits, labels)
        loss_total = self.itc_value + self.itm_value
        return loss_total


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.3, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, image, text):
        scores = self.sim(image, text)
        pos_scores = scores.diag().view(-1, 1)
        d1 = pos_scores.expand_as(scores)
        d2 = pos_scores.t().expand_as(scores)
        
        cost_s = (self.margin + scores - d1).clamp(min=0)

        cost_im = (self.margin + scores - d2).clamp(min=0)
        # clear diagonals
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s = cost_s.masked_fill_(mask.cuda(), 0)
        cost_im = cost_im.masked_fill_(mask.cuda(), 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()