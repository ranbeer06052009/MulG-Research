import torch
import time
from torch import nn
from evaluation.performance import accuracy, f1_score, AUPRC, eval_affect
from evaluation.complexity import all_in_one_train, all_in_one_test
from utils import format_time

softmax = nn.Softmax(dim=1)


#############################################
# MMDL Wrapper (UNCHANGED, SAFE)
#############################################
class MMDL(nn.Module):
    """
    Multimodal Deep Learning wrapper
    """

    def __init__(self, encoders, fusion, head):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.reps = None
        self.fuseout = None

    def forward(self, modalities):
        """
        modalities: list of tensors [X_v, X_a, X_l]
        """
        outs = [enc(mod) for enc, mod in zip(self.encoders, modalities)]
        self.reps = outs
        fused = self.fuse(outs)
        self.fuseout = fused
        return self.head(fused)


#############################################
# TRAIN FUNCTION (CLASSIFICATION ONLY)
#############################################
def train(
    encoders,
    fusion,
    head,
    train_dataloader,
    valid_dataloader,
    total_epochs,
    lr=1e-3,
    weight_decay=0.0,
    optimtype=torch.optim.Adam,
    save="best.pt",
    early_stop=False,
    clip_val=8,
    track_complexity=True,
):
    """
    MulG-compatible supervised training loop
    (Binary / Multiclass classification ONLY)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMDL(encoders, fusion, head).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimtype(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_losses = []
    valid_losses = []

    best_acc = 0.0
    patience = 0

    def _trainprocess():
        nonlocal best_acc, patience

        for epoch in range(total_epochs):
            print(f"Epoch [{epoch+1}/{total_epochs}]")

            #################################
            # TRAIN
            #################################
            model.train()
            total_loss = 0.0
            total_samples = 0

            for j in train_dataloader:
                optimizer.zero_grad()

                # ---- INPUT ----
                modalities = [m.float().to(device) for m in j[0]]
                labels = j[-1].view(-1).long().to(device)
                assert labels.min() >= 0 and labels.max() < 2, \
                f"Invalid labels: min={labels.min()}, max={labels.max()}"

                out = model(modalities)
                loss = criterion(out, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            train_loss = total_loss / total_samples
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            #################################
            # VALIDATION
            #################################
            model.eval()
            total_loss = 0.0
            preds, trues = [], []

            with torch.no_grad():
                for j in valid_dataloader:
                    modalities = [m.float().to(device) for m in j[0]]
                    labels = j[-1].view(-1).long().to(device)
                    assert labels.min() >= 0 and labels.max() < 2, \
                    f"Invalid labels: min={labels.min()}, max={labels.max()}"



                    out = model(modalities)
                    loss = criterion(out, labels)

                    total_loss += loss.item() * labels.size(0)
                    preds.append(torch.argmax(out, dim=1))
                    trues.append(labels)

            preds = torch.cat(preds)
            trues = torch.cat(trues)

            val_loss = total_loss / trues.size(0)
            val_acc = accuracy(trues, preds)

            valid_losses.append(val_loss)

            print(
                f"Valid Loss: {val_loss:.4f} | "
                f"Accuracy: {val_acc*100:.2f}%"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                torch.save(model, save)
                print("✔ Saved Best Model")
            else:
                patience += 1

            if early_stop and patience > 7:
                print("Early stopping triggered.")
                break

            print("")

    if track_complexity:
        all_in_one_train(_trainprocess, [model])
    else:
        _trainprocess()

    return train_losses, valid_losses

def single_test(
    model,
    test_dataloader,
    task="classification",
    auprc=False
):
    """
    MulG-compatible testing (classification)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds, trues = [], []
    pts = []

    with torch.no_grad():
        for j in test_dataloader:
            modalities = [m.float().to(device) for m in j[0]]
            labels = j[-1].view(-1).long().to(device)
            assert labels.min() >= 0 and labels.max() < 2, \
            f"Invalid labels: min={labels.min()}, max={labels.max()}"

            out = model(modalities)

            pred = torch.argmax(out, dim=1)
            preds.append(pred)
            trues.append(labels)

            if auprc:
                sm = softmax(out)
                pts += [
                    (sm[i][1].item(), labels[i].item())
                    for i in range(labels.size(0))
                ]

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    acc = accuracy(trues, preds)
    print(f"Test Accuracy: {acc*100:.2f}%")

    if auprc:
        print("AUPRC:", AUPRC(pts))

    return {"Accuracy": acc}


def test(model, test_dataloader, auprc=False):
    """
    Wrapper for complexity tracking
    """

    def _testprocess():
        single_test(model, test_dataloader, auprc=auprc)

    all_in_one_test(_testprocess, [model])
