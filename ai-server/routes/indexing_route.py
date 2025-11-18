from fastapi import APIRouter, status

from services.semantic_search_service import SemanticSearchService
from models.request_schemas import DocumentEmbeddingRequest
from models.respons_schemas import VectorStoreOperationResponse

router = APIRouter(prefix="/indexing", tags=["indexing"])


def get_semantic_search_service() -> SemanticSearchService:
    return SemanticSearchService()


@router.post(
    "/documents/process",
    response_model=VectorStoreOperationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def process_documents(
    request: DocumentEmbeddingRequest,
):
    service = get_semantic_search_service()
    return await service.create_vector_store(
        request.account_id, int(request.bucket_id), request.documents
    )
