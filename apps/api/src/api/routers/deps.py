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
    qdrant = request.app.state.qdrant
    name = version or settings.experiment_id
    collection = collection_for_version(name)
    if not qdrant.collection_exists(collection):
        raise HTTPException(status_code=404, detail=f"Unknown DB version: {name}")
    return collection
