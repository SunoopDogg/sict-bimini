from unittest.mock import MagicMock

from qdrant_client.models import Filter

from api.bim.predict.retriever import NeighborRetriever
from api.bim.predict.schemas import Neighbor


def _scored_point(score: float, payload: dict):
    pt = MagicMock()
    pt.score = score
    pt.payload = payload
    return pt


class TestNeighborRetriever:
    def test_search_passes_collection_and_vector(self):
        mock_client = MagicMock()
        mock_client.search.return_value = []

        retriever = NeighborRetriever(mock_client, collection="bim__test")
        retriever.search([0.1, 0.2, 0.3], code_field="kbims_code", k=10)

        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["collection_name"] == "bim__test"
        assert kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert kwargs["limit"] == 10

    def test_search_applies_code_field_filter(self):
        mock_client = MagicMock()
        mock_client.search.return_value = []

        retriever = NeighborRetriever(mock_client, collection="bim__test")
        retriever.search([0.0], code_field="kbims_code", k=5)

        qfilter: Filter = mock_client.search.call_args.kwargs["query_filter"]
        dumped = qfilter.model_dump(by_alias=True)
        assert dumped["must"][0]["key"] == "kbims_code"
        assert dumped["must"][0]["match"]["except"] == [""]

    def test_search_maps_points_to_neighbors(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [
            _scored_point(
                0.9,
                {
                    "stable_id": "abc",
                    "kbims_code": "KM001",
                    "pps_code": "",
                    "ifc_type": "IfcColumn",
                    "category": "건축",
                },
            ),
            _scored_point(
                0.8,
                {
                    "stable_id": "def",
                    "kbims_code": "KM002",
                    "pps_code": "A-1",
                    "ifc_type": "IfcBeam",
                    "category": "건축",
                },
            ),
        ]

        retriever = NeighborRetriever(mock_client, collection="bim__test")
        neighbors = retriever.search([0.0], code_field="kbims_code", k=10)

        assert len(neighbors) == 2
        assert isinstance(neighbors[0], Neighbor)
        assert neighbors[0].stable_id == "abc"
        assert neighbors[0].score == 0.9
        assert neighbors[0].kbims_code == "KM001"
        assert neighbors[1].pps_code == "A-1"

    def test_search_uses_pps_field_for_pps_target(self):
        mock_client = MagicMock()
        mock_client.search.return_value = []

        retriever = NeighborRetriever(mock_client, collection="bim__test")
        retriever.search([0.0], code_field="pps_code", k=5)

        qfilter: Filter = mock_client.search.call_args.kwargs["query_filter"]
        assert qfilter.model_dump()["must"][0]["key"] == "pps_code"

    def test_search_returns_empty_list_when_qdrant_returns_nothing(self):
        mock_client = MagicMock()
        mock_client.search.return_value = []

        retriever = NeighborRetriever(mock_client, collection="bim__test")
        assert retriever.search([0.0], code_field="kbims_code", k=10) == []
