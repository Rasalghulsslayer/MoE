import pytest
from training.schedule import get_lr
from training.train_config import TrainConfig
import numpy as np

@pytest.fixture
def schedule():
    return TrainConfig()

def test_schedule_pure(schedule):
    assert get_lr(0, schedule) != 0
    assert abs(get_lr(step = schedule.warmup_steps, schedule = schedule) - schedule.base_lr) < 1e-10

@pytest.mark.parametrize("step_first, step_second", list(zip(np.arange(0, 200, 10),(np.arange(1, 201, 10)))))
def test_schedule_mouvement(schedule, step_first, step_second):
    if step_second < schedule.warmup_steps:
        assert get_lr(step_first, schedule) < get_lr(step_second, schedule)
    elif step_first > schedule.warmup_steps:
        assert get_lr(step_first, schedule) > get_lr(step_second, schedule)