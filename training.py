import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


from model import CustomEfficientNet
from data import get_data_loaders


def train_one_epoch(train_dl, model, criterion, optimizer):

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    training_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in tqdm(
        enumerate(train_dl),
        desc="Training",
        total=len(train_dl),
        leave=True,
        ncols=80,
    ):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

        _, prediction = torch.max(output, 1)

        correct += ((prediction == labels).sum()).item()

        total += images.size(0)

    return training_loss/len(train_dl), (correct / total)*100


def test_one_epoch(test_dl, model, criterion):

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    test_loss, correct, total = 0.0, 0.0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(
            enumerate(test_dl),
            desc="Testing",
            total=len(test_dl),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()

            _, prediction = torch.max(output, 1)

            correct += ((prediction == labels).sum()).item()

            total += images.size(0)

    return test_loss/len(test_dl), (correct / total)*100


def validate_model(valid_dl, model, criterion):

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    eval_loss, correct, total = 0.0, 0.0, 0

    for batch_idx, (images, labels) in tqdm(
        enumerate(valid_dl),
        desc="Validation",
        total=len(valid_dl),
        leave=True,
        ncols=80,
    ):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        output = model(images)
        loss = criterion(output, labels)
        eval_loss += loss.item()

        _, prediction = torch.max(output, 1)

        correct += ((prediction == labels).sum()).item()

        total += images.size(0)

    return eval_loss/len(valid_dl), (correct / total) * 100


def model_training(model, optimizer, criterion, data_loaders, epochs, schedular, saving_path):

    train_losses, valid_losses = [], []

    valid_loss_min, _ = validate_model(
        valid_dl=data_loaders['valid'],
        model=model,
        criterion=criterion,
    )

    for epoch in range(epochs):

        training_loss, training_accuracy = train_one_epoch(
            train_dl=data_loaders['train'],
            model=model,
            criterion=criterion,
            optimizer=optimizer
        )

        train_losses.append(training_loss)

        valid_loss, valid_accuracy = validate_model(
            valid_dl=data_loaders['valid'],
            model=model,
            criterion=criterion,
        )

        valid_losses.append(valid_loss)

        schedular.step()

        if ((valid_loss_min-valid_loss)/valid_loss_min) > 0.01:
            print(
                f"New Minimum validation loss : {valid_loss}.\nSaving model...")

            torch.save(model.state_dict(), saving_path)
            valid_loss_min = valid_loss

        print(f"Epoch {epoch+1}: \n\tTraining_Loss = {training_loss}, Training_accuracy = {training_accuracy}. \n\tValidation_Loss = {valid_loss}, Validation_accuracy = {valid_accuracy}")

    return train_losses, valid_losses


def get_loss_function():
    return torch.nn.CrossEntropyLoss()


def get_optimizer(model, learning_rate=0.001):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_scheduler(optimizer, gamma=0.1):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def plot_loss(train_loss, valid_loss):

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    epochs = 10
    model_saving_path = "./model.pth"
    model = CustomEfficientNet()
    optimizer = get_optimizer(model)
    criterion = get_loss_function()
    schedular = get_scheduler(optimizer)

    data_loaders = get_data_loaders()

    training_loss, valid_loss = model_training(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loaders=data_loaders,
        epochs=epochs,
        schedular=schedular,
        saving_path=model_saving_path
    )

    plot_loss(training_loss, valid_loss)
    print("Training completed and model saved.")

    testing_loss, testing_accuracy = test_one_epoch(
        test_dl=data_loaders['test'],
        model=model,
        criterion=criterion
    )
    print(
        f"Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy}")
    print("Testing completed.")
    print("Model training and testing completed successfully.")
