import pytest

from energy.config import Settings
from energy.data import generate
from energy.training import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    patch = pytest.MonkeyPatch()
    patch.setenv("ENERGY_RUNTIME", str(root))
    patch.setenv("MLFLOW_TRACKING_URI", (root / "mlruns").as_uri())
    settings = Settings(hours=1440, n_estimators=12)
    frame = generate(settings.hours, faults=False)
    report = train(frame, settings, root / "output")
    yield settings, root, report
    patch.undo()
