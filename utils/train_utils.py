import torch 

def accuracy_score(targets: torch.Tensor, predictions: torch.Tensor) -> float: 

    predictions = torch.argmax(predictions, dim = -1)

    classified = torch.sum(torch.eq(targets, predictions).int(), dim = -1)

    return sum(classified) 