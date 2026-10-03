"""Offline CPU demo: classify synthetic vertical versus horizontal bars."""
import torch
from torch import nn


class BarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.features(x).mean((-2, -1)))


def bars(count=128, seed=7):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (count,), generator=generator)
    images = torch.rand(count, 1, 16, 16, generator=generator) * 0.1
    for i, label in enumerate(labels):
        position = int(torch.randint(3, 12, (), generator=generator))
        if label == 0:
            images[i, 0, :, position:position + 2] += 0.9
        else:
            images[i, 0, position:position + 2, :] += 0.9
    return images, labels


def trained_demo(steps=80):
    # Preserve caller RNG state; use no external datasets or checkpoints.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        model = BarNet()
        images, labels = bars()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval()
    return model


def main():
    from autoexplain import Inspector, concept_direction, steer
    torch.set_num_threads(2)
    model = trained_demo()
    images, labels = bars(64, seed=99)
    inspector = Inspector(model)
    with torch.no_grad():
        logits = model(images)
        accuracy = (logits.argmax(1) == labels).float().mean().item()
        features = model.features(images).mean((-2, -1))
        direction = concept_direction(features[labels == 0], features[labels == 1])
    heatmaps = inspector.gradcam(images)
    with steer(model, "features", direction, feature_axis=1, strength=0.5):
        shifted = model(images).detach()
    print(f"Held-out synthetic accuracy: {accuracy:.1%}")
    print(f"Grad-CAM shape: {tuple(heatmaps.shape)}; finite: {bool(heatmaps.isfinite().all())}")
    print(f"Mean absolute logit change under steering: {(shifted - logits).abs().mean():.4f}")
    for suggestion in inspector.suggest():
        print(f"{suggestion.method}: {suggestion.requirements}")


if __name__ == "__main__":
    main()
