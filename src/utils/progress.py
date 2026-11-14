from copy import copy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from time import perf_counter

from lightning.pytorch.callbacks import ProgressBar
from torch import Tensor
from typing_extensions import override

from .pylogger import RankedLogger


@dataclass
class _ProgressState:
    """Keep batch counts and active time local to one stage and dataloader."""

    stage: str
    total_batch: int | float
    current_batch: int = 0
    initial_batch: int = 0
    dataloader_idx: int | None = None
    summary_logged: bool = False
    started_at: float = field(default_factory=lambda: perf_counter())
    paused_at: float | None = None
    paused_duration: float = 0
    finished_at: float | None = None


class RankedProgressBar(ProgressBar):
    BAR_FORMAT = "{head}{metrics}{eta}{rate}"

    def __init__(self, refresh_rate: int = 1, rank_zero_only: bool = True) -> None:
        super().__init__()
        if isinstance(refresh_rate, bool) or not isinstance(refresh_rate, int):
            raise TypeError("refresh_rate must be an integer.")
        if refresh_rate < 0:
            raise ValueError("refresh_rate must be non-negative.")
        self.refresh_rate = refresh_rate
        self.rank_zero_only = rank_zero_only
        self._log = RankedLogger(__name__, rank_zero_only=rank_zero_only)
        self._enabled = True
        self._train_progress: _ProgressState | None = None
        self._eval_progress: dict[tuple[str, int], _ProgressState] = {}
        self._train_elapsed = 0.0
        self._train_completed_batch = 0
        self._train_initial_step: int | None = None

    @override
    def setup(self, trainer, pl_module, stage):
        enabled = self._enabled
        super().setup(trainer, pl_module, stage)
        if not self.rank_zero_only:
            self._enabled = enabled
        self._train_progress = None
        self._eval_progress.clear()
        self._train_elapsed = 0.0
        self._train_completed_batch = 0
        self._train_initial_step = None

    @property
    def is_enabled(self) -> bool:
        """Whether this process should emit progress logs."""
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        """Whether progress logging is disabled."""
        return not self.is_enabled

    @override
    def enable(self) -> None:
        """Enable logging for Lightning utilities that temporarily disable progress."""
        self._enabled = True

    @override
    def disable(self) -> None:
        """Disable progress logging, including on nonzero ranks during setup."""
        self._enabled = False

    @staticmethod
    def _is_last_batch(trainer, state) -> bool:
        can_stop = trainer.current_epoch >= (
            trainer.min_epochs or 0
        ) and trainer.global_step >= (trainer.min_steps or 0)
        return (
            state.current_batch == state.total_batch
            or trainer.is_last_batch
            or (trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps)
            or (trainer.should_stop and can_stop)
        )

    @staticmethod
    def _format_eta(seconds: float | None) -> str:
        """Format elapsed or remaining seconds in the familiar MM:SS form."""
        if seconds is None:
            return "?"
        minutes, seconds = divmod(int(max(seconds, 0)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _format_metric(key, value) -> str:
        key = key.removeprefix("train/").removeprefix("val/").removesuffix("_step")
        if isinstance(value, Tensor) and value.numel() == 1:
            value = value.detach().item()
        return f"{key}={value:.04g}" if isinstance(value, Real) else f"{key}={value}"

    @staticmethod
    def _elapsed(state: _ProgressState) -> float:
        """Return active time, freezing completed stages and excluding validation pauses."""
        now = state.finished_at
        if now is None:
            now = state.paused_at if state.paused_at is not None else perf_counter()
        return max(now - state.started_at - state.paused_duration, 0)

    def _remaining_training_time(self, trainer, state, elapsed, rate) -> float | None:
        """Estimate the remaining fit using the earliest configured training limit."""
        estimates = []
        if (
            trainer.max_epochs is not None
            and trainer.max_epochs > 0
            and isfinite(state.total_batch)
            and rate is not None
        ):
            batches = max(state.total_batch - state.current_batch, 0)
            if state.finished_at is not None:
                batches = 0
            epochs_left = max(trainer.max_epochs - trainer.current_epoch - 1, 0)
            estimates.append((batches + epochs_left * state.total_batch) / rate)
        if trainer.max_steps > 0:
            steps_left = max(trainer.max_steps - trainer.global_step, 0)
            if steps_left == 0:
                return 0
            if self._train_initial_step is not None:
                steps = trainer.global_step - self._train_initial_step
                if steps > 0 and elapsed > 0:
                    estimates.append(steps_left * elapsed / steps)
        return min(estimates) if estimates else None

    def _log_progress(
        self,
        trainer,
        pl_module,
        state: _ProgressState,
        include_metrics=True,
        on_step=True,
    ):
        if not self.is_enabled or (self.rank_zero_only and not trainer.is_global_zero):
            return

        metrics = (
            self.get_metrics(trainer, pl_module, on_step=on_step)
            if include_metrics
            else {}
        )

        stage = state.stage
        current_epoch = trainer.current_epoch + 1
        total_batch = state.total_batch if state.total_batch != float("inf") else "?"
        head = f"[Epoch {current_epoch} Iter {state.current_batch}/{total_batch}] "
        if state.dataloader_idx is not None:
            head += f"{stage.capitalize()} DataLoader {state.dataloader_idx}, "

        metrics_text = ", ".join(
            self._format_metric(key, value) for key, value in sorted(metrics.items())
        )

        elapsed = self._elapsed(state)
        completed = state.current_batch - state.initial_batch
        if stage == "train":
            elapsed += self._train_elapsed
            completed += self._train_completed_batch
        rate = completed / elapsed if completed > 0 and elapsed > 0 else None
        remaining = None
        if stage == "train":
            remaining = self._remaining_training_time(trainer, state, elapsed, rate)
        elif not on_step or state.current_batch >= state.total_batch:
            remaining = 0
        elif isfinite(state.total_batch) and rate is not None:
            remaining = (state.total_batch - state.current_batch) / rate
        eta = f"ETA {self._format_eta(elapsed)}<{self._format_eta(remaining)}, "
        rate_text = f"{rate:.2f}it/s" if rate is not None else "?it/s"

        self._log.info(
            self.BAR_FORMAT.format(
                head=head,
                metrics=f"{metrics_text}, " if metrics_text else "",
                eta=eta,
                rate=rate_text,
            )
        )

    def _log_batch(self, trainer, pl_module, state, batch_idx, include_metrics=True):
        state.current_batch = batch_idx + 1
        last_batch = state.current_batch == state.total_batch
        if trainer.training:
            last_batch = self._is_last_batch(trainer, state)
        if self.refresh_rate > 0 and (
            state.current_batch % self.refresh_rate == 0 or last_batch
        ):
            self._log_progress(trainer, pl_module, state, include_metrics)
        return last_batch

    def _finish_stage(self, trainer, pl_module, state, include_metrics=True):
        if (
            state is None
            or state.current_batch <= state.initial_batch
            or state.summary_logged
        ):
            return
        self._log_progress(
            trainer,
            pl_module,
            state,
            include_metrics=include_metrics,
            on_step=False,
        )
        state.summary_logged = True

    def _finish_training(self, trainer, pl_module):
        if (
            self._train_progress is not None
            and self._train_progress.finished_at is None
        ):
            self._train_progress.finished_at = perf_counter()
        self._finish_stage(trainer, pl_module, self._train_progress)

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        previous = self._train_progress
        if previous is not None:
            self._train_elapsed += self._elapsed(previous)
            self._train_completed_batch += (
                previous.current_batch - previous.initial_batch
            )
        if self._train_initial_step is None:
            self._train_initial_step = trainer.global_step
        self._train_progress = _ProgressState("train", trainer.num_training_batches)

    @override
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Initialize lazily because mid-epoch restores skip on_train_epoch_start."""
        if self._train_progress is None:
            self.on_train_epoch_start(trainer, pl_module)
        if self._train_progress.current_batch == 0 and batch_idx > 0:
            self._train_progress.current_batch = batch_idx
            self._train_progress.initial_batch = batch_idx
            self._train_progress.started_at = perf_counter()

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._log_batch(trainer, pl_module, self._train_progress, batch_idx):
            self._finish_training(trainer, pl_module)

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        self._finish_training(trainer, pl_module)

    def _start_evaluation(self, stage):
        self.reset_dataloader_idx_tracker()
        self._eval_progress = {
            key: state for key, state in self._eval_progress.items() if key[0] != stage
        }

    def _evaluation_batch_start(self, stage, dataloader_idx, batch_idx):
        self.has_dataloader_changed(dataloader_idx)
        key = (stage, dataloader_idx)
        if key not in self._eval_progress:
            total_batch = getattr(self, f"total_{stage}_batches_current_dataloader")
            self._eval_progress[key] = _ProgressState(
                stage,
                total_batch,
                dataloader_idx=dataloader_idx,
                current_batch=batch_idx,
                initial_batch=batch_idx,
            )

    def _finish_evaluation(self, trainer, pl_module, stage):
        states = [
            state
            for key, state in self._eval_progress.items()
            if key[0] == stage and not state.summary_logged
        ]
        if states:
            state = states[0]
            if len(states) > 1:
                state = _ProgressState(
                    stage,
                    sum(item.total_batch for item in states),
                    current_batch=sum(item.current_batch for item in states),
                    initial_batch=sum(item.initial_batch for item in states),
                    started_at=min(item.started_at for item in states),
                )
            self._finish_stage(
                trainer, pl_module, state, include_metrics=stage != "predict"
            )
            for item in states:
                item.summary_logged = True
        self.reset_dataloader_idx_tracker()

    @override
    def on_validation_start(self, trainer, pl_module):
        train = self._train_progress
        if train is not None and not train.summary_logged and train.paused_at is None:
            train.paused_at = perf_counter()
        self._start_evaluation("val")

    @override
    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not trainer.sanity_checking:
            self._evaluation_batch_start("val", dataloader_idx, batch_idx)

    @override
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not trainer.sanity_checking:
            self._log_batch(
                trainer,
                pl_module,
                self._eval_progress[("val", dataloader_idx)],
                batch_idx,
            )

    @override
    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._finish_evaluation(trainer, pl_module, "val")
        train = self._train_progress
        if train is not None and train.paused_at is not None:
            train.paused_duration += perf_counter() - train.paused_at
            train.paused_at = None

    @override
    def on_test_start(self, trainer, pl_module):
        self._start_evaluation("test")

    @override
    def on_test_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._evaluation_batch_start("test", dataloader_idx, batch_idx)

    @override
    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._log_batch(
            trainer,
            pl_module,
            self._eval_progress[("test", dataloader_idx)],
            batch_idx,
        )

    @override
    def on_test_end(self, trainer, pl_module):
        self._finish_evaluation(trainer, pl_module, "test")

    @override
    def on_predict_start(self, trainer, pl_module):
        self._start_evaluation("predict")

    @override
    def on_predict_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._evaluation_batch_start("predict", dataloader_idx, batch_idx)

    @override
    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._log_batch(
            trainer,
            pl_module,
            self._eval_progress[("predict", dataloader_idx)],
            batch_idx,
            include_metrics=False,
        )

    @override
    def on_predict_end(self, trainer, pl_module):
        self._finish_evaluation(trainer, pl_module, "predict")

    @override
    def print(self, *args, sep=" ", **kwargs):
        self._log.info(sep.join(map(str, args)))

    @override
    def get_metrics(self, trainer, pl_module, *, on_step=True) -> dict:
        """Read current results, without the trainer's persistent progress cache."""
        results = trainer._results
        if results is None:
            return {}
        if on_step:
            return results.metrics(on_step=True)["pbar"]

        metrics = {}
        view = copy(results)
        dataloader_indices = dict.fromkeys(
            result.meta.dataloader_idx
            for result in results.values()
            if not result.has_reset
        )
        for dataloader_idx in dataloader_indices:
            view.dataloader_idx = dataloader_idx
            metrics.update(view.metrics(on_step=False)["pbar"])
        return metrics
