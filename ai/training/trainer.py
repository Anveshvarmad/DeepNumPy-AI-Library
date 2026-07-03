from ..core import no_grad


class Trainer:
    def __init__(self, model, loss_fn, optimizer, metrics=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = self._normalize_metrics(metrics)

    def _normalize_metrics(self, metrics):
        if metrics is None:
            return []

        if isinstance(metrics, dict):
            return list(metrics.items())

        normalized = []

        for metric in metrics:
            name = getattr(metric, "__name__", metric.__class__.__name__)
            normalized.append((name, metric))

        return normalized

    def fit(self, train_loader, epochs=1, val_loader=None, log_every=1):
        history = {
            "train_loss": [],
        }

        for metric_name, _ in self.metrics:
            history[f"train_{metric_name}"] = []

        if val_loader is not None:
            history["val_loss"] = []

            for metric_name, _ in self.metrics:
                history[f"val_{metric_name}"] = []

        for epoch in range(1, epochs + 1):
            train_results = self.train_epoch(train_loader)

            history["train_loss"].append(train_results["loss"])

            for metric_name, _ in self.metrics:
                history[f"train_{metric_name}"].append(train_results[metric_name])

            message = f"epoch={epoch}, train_loss={train_results['loss']:.6f}"

            for metric_name, _ in self.metrics:
                message += f", train_{metric_name}={train_results[metric_name]:.4f}"

            if val_loader is not None:
                val_results = self.evaluate(val_loader)

                history["val_loss"].append(val_results["loss"])

                for metric_name, _ in self.metrics:
                    history[f"val_{metric_name}"].append(val_results[metric_name])

                message += f", val_loss={val_results['loss']:.6f}"

                for metric_name, _ in self.metrics:
                    message += f", val_{metric_name}={val_results[metric_name]:.4f}"

            if log_every is not None and epoch % log_every == 0:
                print(message)

        return history

    def train_epoch(self, data_loader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        metric_totals = {
            metric_name: 0.0
            for metric_name, _ in self.metrics
        }

        for batch in data_loader:
            x, y = batch

            prediction = self.model(x)
            loss = self.loss_fn(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = x.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for metric_name, metric_fn in self.metrics:
                metric_totals[metric_name] += metric_fn(prediction, y) * batch_size

        results = {
            "loss": total_loss / total_samples
        }

        for metric_name in metric_totals:
            results[metric_name] = metric_totals[metric_name] / total_samples

        return results

    def evaluate(self, data_loader):
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        metric_totals = {
            metric_name: 0.0
            for metric_name, _ in self.metrics
        }

        with no_grad():
            for batch in data_loader:
                x, y = batch

                prediction = self.model(x)
                loss = self.loss_fn(prediction, y)

                batch_size = x.shape[0]

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                for metric_name, metric_fn in self.metrics:
                    metric_totals[metric_name] += metric_fn(prediction, y) * batch_size

        if was_training:
            self.model.train()

        results = {
            "loss": total_loss / total_samples
        }

        for metric_name in metric_totals:
            results[metric_name] = metric_totals[metric_name] / total_samples

        return results
