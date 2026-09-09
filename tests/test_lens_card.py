import json

import pytest
from pydantic import ValidationError

from lenscraft.schema import LensCard


def test_defaults_match_deeplensesim():
    card = LensCard(n_images=10, substructure="none")
    assert card.halo_mass == 1e12
    assert card.redshift_lens == 0.5
    assert card.redshift_source == 1.0
    assert card.instrument == "euclid"
    assert card.image_size == 64
    assert card.backend == "lenstronomy"
    assert card.seed is None


@pytest.mark.parametrize(
    "bad",
    [
        {"n_images": 0, "substructure": "none"},
        {"n_images": 100_001, "substructure": "none"},
        {"n_images": 1, "substructure": "blob"},
        {"n_images": 1, "substructure": "vortex", "substructure_mass_fraction": 1.5},
        {"n_images": 1, "substructure": "vortex", "substructure_mass_fraction": 0.0},
        {"n_images": 1, "substructure": "none", "redshift_lens": 1.0, "redshift_source": 0.5},
        {"n_images": 1, "substructure": "subhalo", "subhalo_mass_min": 1e10, "subhalo_mass_max": 1e6},
        {"n_images": 1, "substructure": "none", "backend": "madgraph"},
        {"n_images": 1, "substructure": "none", "not_a_field": 1},
    ],
)
def test_rejects_invalid_cards(bad):
    with pytest.raises(ValidationError):
        LensCard.model_validate(bad)


def test_substructure_mass():
    assert LensCard(n_images=1, substructure="none").substructure_mass == 0.0
    card = LensCard(n_images=1, substructure="vortex", halo_mass=1e12, substructure_mass_fraction=0.03)
    assert card.substructure_mass == pytest.approx(3e10)


def test_json_round_trip():
    card = LensCard(n_images=5, substructure="subhalo", seed=42, axion_mass=None)
    again = LensCard.model_validate_json(card.model_dump_json())
    assert again == card
    assert json.loads(card.model_dump_json())["substructure"] == "subhalo"


def test_frozen():
    card = LensCard(n_images=1, substructure="none")
    with pytest.raises(ValidationError):
        card.n_images = 2  # type: ignore[misc]


def test_schema_is_llm_legible():
    schema = LensCard.model_json_schema()
    assert schema["properties"]["substructure"]["enum"] == ["none", "subhalo", "vortex"]
    assert "description" in schema["properties"]["substructure_mass_fraction"]
