import torch

def train_gait_ensemble(model, train_loader, epochs=15, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # One optimizer now handles all 7 branches automatically
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # preds shape: (batch_size, 7, 2)
            preds = model(x)
            
            # Unsqueeze y and mask to (batch_size, 1, 2) so they broadcast across all 7 branches
            y_expanded = y.unsqueeze(1)
            mask_expanded = mask.unsqueeze(1)
            
            # Compute MSE across all branches simultaneously
            se = (preds - y_expanded) ** 2 * mask_expanded
            
            # Sum the error and divide by the number of valid labels 
            # (multiplying mask sum by 7 because there are 7 branches making predictions)
            denom = mask_expanded.sum() * model.members.__len__()
            loss = se.sum() / torch.clamp(denom, min=1.0)
            
            loss.backward()
            optimizer.step()
            
    return model