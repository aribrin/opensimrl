# OpenSimRL Tutorials

## Basic Training
```python
from opensimrl import train

rewards = train()
```

## Training with Logging
```python
from opensimrl import train_with_logging

train_with_logging(project_name="MyExperiment")
```

## Running Tests
```python
import pytest

pytest.main(["tests/"])
```

For more examples, see the notebooks directory.
