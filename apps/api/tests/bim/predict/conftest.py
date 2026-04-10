"""Shared fixtures for predict tests."""
from __future__ import annotations

import pytest

from api.bim.schemas import BIMAttribute


@pytest.fixture
def sample_attribute() -> BIMAttribute:
    return BIMAttribute(
        ifc_type="IfcColumn",
        category="건축",
        family_name="RC기둥",
        family="기둥",
        type="T1",
        type_id="X",
        kbims_code="",
        pps_code="",
    )
