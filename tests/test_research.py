import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autoexplain.research import CodexPlanner, METHODS, run_experiment, validate_plan


def plan(methods=None):
    return {"question": "Which features affect the class score?", "hypothesis": "Bright pixels contribute.",
            "methods": methods or ["input_gradients"], "target": 0, "layer": "0", "limitations": []}


def test_codex_command_and_validated_response(monkeypatch):
    def fake(command, **kwargs):
        assert command[command.index('--model') + 1] == 'gpt-6-luna'
        assert 'model_reasoning_effort="high"' in command
        assert '--ignore-user-config' in command and '--ephemeral' in command
        assert command[command.index('--sandbox') + 1] == 'read-only'
        assert kwargs['cwd'] != str(Path.cwd())
        assert 'Do not use tools' in kwargs['input']
        Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps(plan()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', fake)
    assert CodexPlanner().propose('Question', {'task': 'public demo'}) == plan()


@pytest.mark.parametrize('failure', ['exit', 'timeout', 'invalid'])
def test_planner_fails_without_retry(monkeypatch, failure):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        if failure == 'invalid':
            Path(command[command.index('--output-last-message') + 1]).write_text('{}')
        return SimpleNamespace(returncode=1 if failure == 'exit' else 0)
    monkeypatch.setattr(subprocess, 'run', fake)
    with pytest.raises((ValueError, RuntimeError)):
        CodexPlanner().propose('Question', {})
    assert len(calls) == 1


@pytest.mark.parametrize('change', [{'methods': ['exec']}, {'target': True}, {'methods': ['gradcam', 'gradcam']}, {'code': 'print(1)'}])
def test_invalid_plan(change):
    proposal = plan()
    proposal.update(change)
    with pytest.raises(ValueError):
        validate_plan(proposal)


def test_all_methods_local_and_artifacts(tmp_path):
    torch.set_num_threads(2)
    model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    x = torch.arange(16.).reshape(1, 1, 4, 4) / 16
    report = run_experiment(model, x, plan(list(METHODS)), tmp_path / 'run')
    assert report['status'] == 'completed' and len(report['results']) == 6
    assert report['hypothesis_status'] == 'not_adjudicated'
    assert set(torch.load(tmp_path / 'run/attributions.pt', weights_only=True)) == set(METHODS)
    assert json.loads((tmp_path / 'run/report.json').read_text())['status'] == 'completed'
    assert not model[0]._forward_hooks and model.training
    with pytest.raises(FileExistsError):
        run_experiment(model, x, plan(), tmp_path / 'run')


def test_budget_and_failed_report(tmp_path):
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        run_experiment(model, torch.zeros(5, 2), plan(), tmp_path / 'budget')
    assert not (tmp_path / 'budget').exists()
    proposal = plan()
    proposal['target'] = 99
    with pytest.raises(RuntimeError):
        run_experiment(model, torch.ones(1, 2), proposal, tmp_path / 'failure')
    assert json.loads((tmp_path / 'failure/report.json').read_text())['status'] == 'failed'
