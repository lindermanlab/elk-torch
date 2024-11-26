import wandb
import torch
import torch.nn as nn
from elk-torch.data.dataloaders.basic import load_sequential_mnist
from elk-torch.models.minrnn import MinRNNClassifier
from tqdm import tqdm

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of seq_mnist epochs to train for",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10,
        help="Number of quasi-DEER iterations to use",
    )
    parser.add_argument(
        "--entity", type=str, default="xavier_gonzalez", help="wandb entity"
    )
    args = parser.parse_args()

    # set up wandb
    wandb.init(project="nonlinear_ssm", entity=args.entity)

    # Set hyperparameters
    input_size = 1
    hidden_size = args.hidden_size  # 256
    num_epochs = args.num_epochs  # 100
    batch_size = 256  # 256
    learning_rate = 0.001
    num_iters = args.num_iters  # 10
    print(f"Using {num_iters} quasi-DEER iterations")

    # Load datasets
    train_loader = load_sequential_mnist("train", batch_size)
    test_loader = load_sequential_mnist("test", batch_size)

    # train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinRNNClassifier(hidden_dim=128, input_dim=1, num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(num_epochs)):
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            x = x.permute(2, 0, 1)  # (T,B,D)
            optimizer.zero_grad()
            logits = model(x, parallel=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # eval loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in tqdm(test_loader):
                x = x.to(device)
                y = y.to(device)
                x = x.permute(2, 0, 1)  # (T,B,D)
                predictions = model.predict(x, parallel=True)
                correct += torch.sum(predictions == y)
                total += len(y)
            print(
                f"Epoch {epoch}, Val Accuracy: {round(100 * (correct/total).item(),2)}%"
            )
            wandb.log({"val_accuracy": round(100 * (correct / total).item(), 2)})
    print("Training complete")
    print(f"Using {num_iters} quasi-DEER iterations")
