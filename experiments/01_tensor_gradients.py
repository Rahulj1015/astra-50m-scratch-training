import torch


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.linspace(-2.0, 2.0, 9, device=device).unsqueeze(1)
    targets = 3.0 * inputs + 2.0

    weight = torch.zeros(1, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    learning_rate = 0.1

    print(f"device={device}")
    print(f"inputs_shape={tuple(inputs.shape)}")
    print(f"targets_shape={tuple(targets.shape)}")

    for step in range(100):
        predictions = inputs * weight + bias
        loss = ((predictions - targets) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad

        weight.grad = None
        bias.grad = None

        if step in (0, 1, 9, 99):
            print(
                f"step={step + 1:03d} loss={loss.item():.6f} "
                f"weight={weight.item():.6f} bias={bias.item():.6f}"
            )

    print(f"final_loss={loss.item():.6f}")
    print(f"expected_weight=3.0 expected_bias=2.0")


if __name__ == "__main__":
    main()