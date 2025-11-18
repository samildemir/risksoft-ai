import logging
import os
from typing import Any, Dict, List, Tuple

from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.load import dumpd, loads
from langchain_community.document_loaders import AmazonTextractPDFLoader
from pydantic import ValidationError

from constants.config import OPENROUTER_GEMINI_FLASH
from constants.env_variables import (
    BUCKET_NAME,
    TEXT_EXTRACT_ACCESS_KEY,
    TEXT_EXTRACT_SECRET_ACCESS,
    OPENAI_API_KEY,
)
from models.schemas import (
    ChatbotUsageLog,
    DocumentSource,
    DocumentQAResponse,
    DocumentAnalysisResponse,
    DOCUMENT_QA_RESPONSE_SCHEMA,
    DOCUMENT_ANALYSIS_RESPONSE_SCHEMA,
)
from prompts.semantic_prompts import (
    build_document_qa_system_prompt,
    build_document_qa_prompt,
    build_document_analysis_system_prompt,
    build_document_analysis_prompt,
)
from services.open_router_service import OpenRouterService
from utils.botoHandler import BotoHandler
from utils.s3Handler import S3Handler
from utils.vector_store import VectorStoreHandler

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    A service for semantic document search and retrieval using LangChain and Pinecone.
    Implements RAG (Retrieval Augmented Generation) pattern for enhanced document Q&A.

    Features:
    - Account-specific document processing and chunking
    - Vector embeddings generation and storage per account
    - Semantic search with conversation history
    - Token usage tracking
    - Async support
    """

    def __init__(self):
        """Initialize the semantic search service with necessary components."""
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Set up all required components and connections."""
        self.openrouter_service = OpenRouterService()

        self.vector_store_handler = VectorStoreHandler()
        self.s3_handler = S3Handler()
        self.textract_boto_handler = BotoHandler(
            "textract",
            aws_access_key_id=TEXT_EXTRACT_ACCESS_KEY,
            aws_secret_access_key=TEXT_EXTRACT_SECRET_ACCESS,
        )
        self.indexing_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=300,
            chunk_overlap=60,
        )

    async def delete_account_vectors(self, account_id: str) -> bool:
        """
        Delete all vectors for a specific account.

        Args:
            account_id: Account identifier

        Returns:
            bool indicating success/failure
        """
        try:
            await self.vector_store_handler.delete_all_for_account(account_id)
            logger.info(f"Successfully deleted vector store for account {account_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector store for account {account_id}: {e}")
            return False

    async def create_vector_store(
        self, account_id: int, bucket_id: int, documents: List[DocumentSource]
    ) -> Dict[str, Any]:
        """
        Create or refresh the vector store for a given account/bucket combination.
        """
        try:
            account_key = str(account_id)
            bucket_key = str(bucket_id)
            documents_list = []

            for document in documents:
                file_name = os.path.basename(document.path)
                file_name_without_extension = file_name.split(".")[0]
                raw_data_path = (
                    f"{account_id}-{bucket_id}/vector_store/raw_data/"
                    f"{file_name_without_extension}.json"
                )

                metadata_base = {
                    "bucket_id": bucket_key,
                    "document_id": str(document.id),
                    "document_name": document.name,
                    "document_path": document.path,
                }

                if not self.s3_handler.check_if_file_exists(raw_data_path):
                    s3_url = f"s3://{BUCKET_NAME}/{document.path}"
                    loader = AmazonTextractPDFLoader(
                        s3_url, client=self.textract_boto_handler.get_client()
                    )
                    extracted_documents = loader.load()
                    self._apply_metadata(extracted_documents, metadata_base)
                    json_documents = dumpd(extracted_documents)
                    documents_list.extend(extracted_documents)
                    self.s3_handler.upload_json(raw_data_path, json_documents)
                else:
                    json_documents = self.s3_handler.s3_get_object(raw_data_path)
                    chain = loads(
                        json_documents, secrets_map={"OPENAI_API_KEY": OPENAI_API_KEY}
                    )
                    self._apply_metadata(chain, metadata_base)
                    documents_list.extend(chain)

            splited_documents = self.indexing_text_splitter.split_documents(
                documents_list
            )

            if not splited_documents:
                logger.warning(
                    "No document chunks generated for account %s bucket %s",
                    account_id,
                    bucket_id,
                )
                return {
                    "success": False,
                    "message": "Belirtilen belgelerden içerik çıkarılamadı.",
                }

            await self.vector_store_handler.delete_by_metadata(
                account_id=account_key, metadata_filter={"bucket_id": bucket_key}
            )

            await self.vector_store_handler.add_texts(
                documents=splited_documents, account_id=account_key
            )

            logger.info(
                "Vector store created successfully for account %s bucket %s",
                account_id,
                bucket_id,
            )
            return {"success": True, "message": "Vector store created successfully"}
        except Exception as e:
            logger.error(f"Document indexing error for account {account_id}: {e}")
            raise

    async def semantic_search(self, query: str, account_id: str) -> str:
        """
        Perform semantic search on indexed documents for a specific account.

        Args:
            query: Search query
            account_id: Account identifier

        Returns:
            Formatted search results
        """
        try:
            results = await self.vector_store_handler.similarity_search(
                query=query, account_id=account_id, k=5
            )

            if not results:
                return "No relevant results found for this account."

            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"Semantic search error for account {account_id}: {e}")
            return "An error occurred during search."

    def _format_search_results(self, results: List[Document]) -> str:
        """Format search results for display."""
        formatted = "Relevant Content:\n"
        for idx, doc in enumerate(results, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            formatted += f"{idx}. [{source}] {content}\n\n"
        return formatted

    async def chat_with_documents(
        self, question: str, account_id: int
    ) -> Tuple[str, ChatbotUsageLog]:
        """
        Chat with documents using RAG pattern.

        Args:
            question: User question
            account_id: Account identifier

        Returns:
            Tuple containing (response content, ChatbotUsageLog)
        """
        try:
            usage_log = ChatbotUsageLog()

            # First, get relevant documents
            results = await self.vector_store_handler.similarity_search(
                query=question, account_id=account_id, k=15
            )

            if not results:
                return "No relevant results found for this account.", usage_log

            # Format the context from search results
            context = self._format_search_results(results)

            # Generate comprehensive answer using OpenRouter
            system_message = build_document_qa_system_prompt()
            prompt = build_document_qa_prompt(question, context)

            response_obj = self.openrouter_service.generate_text(
                prompt=prompt,
                model=OPENROUTER_GEMINI_FLASH,
                temperature=0.2,
                system_message=system_message,
                usage_log=usage_log,
                response_format=DOCUMENT_QA_RESPONSE_SCHEMA,
            )

            try:
                structured = DocumentQAResponse.model_validate_json(
                    response_obj.content.strip()
                )
            except (ValidationError, ValueError) as exc:
                logger.warning("Document QA response parse failed: %s", exc)
                structured = DocumentQAResponse(answer=response_obj.content, key_points=[])

            answer_text = structured.answer.strip()
            if structured.key_points:
                bullet_list = "\n".join(
                    f"- {point.strip()}"
                    for point in structured.key_points
                    if point and point.strip()
                )
                if bullet_list:
                    answer_text = (
                        f"{answer_text}\n\nÖne Çıkan Noktalar:\n{bullet_list}".strip()
                    )

            # Format final response with sources
            final_response = self._format_response_with_sources(answer_text, results)
            return final_response, usage_log

        except Exception as e:
            error_msg = f"Document chat error: {e}"
            logger.error(error_msg)
            return (
                "An error occurred while processing your question.",
                ChatbotUsageLog.create_error_log(error_msg),
            )

    def _format_response_with_sources(
        self, answer: str, sources: List[Document]
    ) -> str:
        """Format the response with source citations."""
        response = f"{answer}\n\nSources:\n"
        for idx, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "Unknown")
            response += f"{idx}. {source}\n"
        return response

    async def advanced_document_chat(
        self,
        question: str,
        account_id: int,
        model: str = "google/gemini-2.0-flash-001",
        temperature: float = 0.2,
    ) -> Tuple[str, ChatbotUsageLog]:
        """
        Advanced document chat with customizable model selection via OpenRouter.

        Args:
            question: User question
            account_id: Account identifier
            model: OpenRouter model to use
            temperature: Response creativity level

        Returns:
            Tuple containing (response content, ChatbotUsageLog)
        """
        try:
            usage_log = ChatbotUsageLog()

            # Get relevant documents using vector search
            results = await self.vector_store_handler.similarity_search(
                query=question,
                account_id=account_id,
                k=10,  # Slightly less for advanced model efficiency
            )

            if not results:
                return "Bu hesap için ilgili sonuç bulunamadı.", usage_log

            # Create rich context from search results
            context_parts = []
            for idx, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get("source", "Bilinmeyen")
                context_parts.append(f"[Kaynak {idx}: {source}]\n{content}")

            context = "\n\n".join(context_parts)

            # Enhanced system message for better responses
            system_message = build_document_analysis_system_prompt()
            prompt = build_document_analysis_prompt(question, context)

            response_obj = self.openrouter_service.generate_text(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_message=system_message,
                usage_log=usage_log,
                response_format=DOCUMENT_ANALYSIS_RESPONSE_SCHEMA,
            )

            try:
                structured = DocumentAnalysisResponse.model_validate_json(
                    response_obj.content.strip()
                )
            except (ValidationError, ValueError) as exc:
                logger.warning("Document analysis response parse failed: %s", exc)
                structured = DocumentAnalysisResponse(
                    answer=response_obj.content, analysis_notes=[]
                )

            answer_text = structured.answer.strip()
            if structured.analysis_notes:
                notes = "\n".join(
                    f"- {note.strip()}"
                    for note in structured.analysis_notes
                    if note and note.strip()
                )
                if notes:
                    answer_text = f"{answer_text}\n\nNotlar:\n{notes}".strip()

            final_response = f"{answer_text}\n\n📚 Kaynaklar:\n"
            for idx, doc in enumerate(results[:5], 1):  # Show top 5 sources
                source = doc.metadata.get("source", "Bilinmeyen")
                final_response += f"{idx}. {source}\n"

            return final_response, usage_log

        except Exception as e:
            error_msg = f"Advanced document chat error: {e}"
            logger.error(error_msg)
            return (
                "Belge analizi sırasında bir hata oluştu.",
                ChatbotUsageLog.create_error_log(error_msg),
            )

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter models for document processing."""
        return self.openrouter_service.get_available_models()

    @staticmethod
    def _apply_metadata(documents: List, metadata: Dict[str, str]) -> None:
        """Attach consistent metadata to every extracted document."""
        for doc in documents:
            if not hasattr(doc, "metadata"):
                continue
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(metadata.copy())
