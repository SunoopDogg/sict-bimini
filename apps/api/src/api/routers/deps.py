from fastapi import HTTPException, Query, Request

from api.bim.versions import collection_for_version


def resolve_collection(
    request: Request,
    version: str | None = Query(default=None),
) -> str:
    """Resolve the ``?version=`` query param to a validated Qdrant collection.

    Omitted → the env default ``experiment_id``. Unknown collection → 404.
    """
    settings = request.app.state.bim
    if version is None:
        # Default collection is built at startup → guaranteed to exist;
        # reuse the single naming source on BIMSettings, skip the probe.
        return settings.collection_name
    collection = collection_for_version(version)
    if not request.app.state.qdrant.collection_exists(collection):
        raise HTTPException(status_code=404, detail=f"Unknown DB version: {version}")
    return collection
