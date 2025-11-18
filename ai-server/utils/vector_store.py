import os
from typing import List, Optional, Dict, Any
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import time

from utils.helper import get_env
import logging
from uuid import uuid4
logger = logging.getLogger(__name__)

class VectorStoreHandler:
    def __init__(self):
        """Initialize the Vector Store Service with Pinecone"""
        self.pinecone_api_key = get_env("PINECONE_API_KEY")
        self.pinecone_environment = get_env("PINECONE_ENVIRONMENT", "gcp-starter")
        self.base_index_name = "document-index-204"  # Sabit index ismi
        
        # Initialize Pinecone with new method
        self.pinecone_client = PineconeClient(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment
        )
        
        # Initialize OpenAI embeddings with the specified model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Cache for vector stores
        self._vector_stores = {}

    def _normalize_account_id(self, account_id: str) -> str:
        """Normalize account identifiers for consistent caching and filtering."""
        return str(account_id)

    def _get_vector_store(self, account_id: str) -> Pinecone:
        """Get or create a vector store for an account"""
        account_key = self._normalize_account_id(account_id)
        if account_key not in self._vector_stores:
            index_name = self.base_index_name
            
            # Create index if it doesn't exist
            existing_indexes = [index_info["name"] for index_info in self.pinecone_client.list_indexes()]
            
            if index_name not in existing_indexes:
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                while not self.pinecone_client.describe_index(index_name).status["ready"]:
                    time.sleep(1)
            
            # Initialize the vector store
            index = self.pinecone_client.Index(index_name)
            self._vector_stores[account_key] = Pinecone(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
        
        return self._vector_stores[account_key]

    async def add_texts(
        self, 
        documents: List[Document], 
        account_id: str,
    ) -> List[str]:
        """
        Add texts to the vector store
        
        Args:
            documents (List[Document]): List of documents to be added
            account_id (str): Account ID for document isolation
            
        Returns:
            List[str]: List of IDs for the added documents
        """
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            
            # Add account_id to metadata of each document
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['account_id'] = account_key
            
            # Add documents to vector store
            uuids = [str(uuid4()) for _ in range(len(documents))]
            ids = vector_store.add_documents(documents=documents, ids=uuids)
            
            # Get stats for the index
            index = self.pinecone_client.Index(self.base_index_name)
            stats = index.describe_index_stats()
            logger.info(f"Total vectors in index: {stats['total_vector_count']}")
            
            return ids
        
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise

    async def similarity_search(
        self,
        query: str,
        account_id: str,
        k: int = 5,
        additional_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store
        
        Args:
            query (str): Query text to search for
            account_id (str): Account ID to filter results
            k (int): Number of results to return
            additional_filter (Optional[Dict[str, Any]]): Additional metadata filter
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            
            # Create filter with account_id
            filter_dict = {"account_id": account_key}
            if additional_filter:
                filter_dict.update(additional_filter)
            
            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            return results
        
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise

    async def delete_documents(self, ids: List[str], account_id: str) -> None:
        """
        Delete documents from the account's vector store
        
        Args:
            ids (List[str]): List of document IDs to delete
            account_id (str): Account ID for verification
        """
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            vector_store.delete(ids)
            logger.info(f"Successfully deleted {len(ids)} documents from vector store for account {account_id}")
        
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            raise

    async def update_document(
        self,
        doc_id: str,
        account_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a document in the account's vector store
        
        Args:
            doc_id (str): ID of the document to update
            account_id (str): Account ID for verification
            text (str): New text content
            metadata (Optional[Dict[str, Any]]): New metadata
        """
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            
            # Delete the old document
            await self.delete_documents([doc_id], account_key)
            
            # Create new document
            if metadata is None:
                metadata = {}
            
            doc = Document(page_content=text, metadata=metadata)
            
            # Add the new document with the same ID
            vector_store.add_documents(documents=[doc], ids=[doc_id])
            logger.info(f"Successfully updated document {doc_id} for account {account_id}")
        
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise

    async def get_account_documents(
        self,
        account_id: str,
        limit: int = 100
    ) -> List[Document]:
        """
        Retrieve all documents from the account's vector store
        
        Args:
            account_id (str): Account ID to filter documents
            limit (int): Maximum number of documents to return
            
        Returns:
            List[Document]: List of documents belonging to the account
        """
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            results = vector_store.similarity_search(
                query="",
                k=limit
            )
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving account documents: {e}")
            raise 

    async def delete_all_for_account(self, account_id: str) -> None:
        """Delete all vectors belonging to a specific account."""
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            vector_store.delete(filter={"account_id": account_key})
            logger.info("Deleted all vectors for account %s", account_key)
        except Exception as e:
            logger.error(f"Error deleting all vectors for account {account_id}: {e}")
            raise

    async def delete_by_metadata(
        self, account_id: str, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """Delete account vectors that match an additional metadata filter."""
        try:
            account_key = self._normalize_account_id(account_id)
            vector_store = self._get_vector_store(account_key)
            filter_dict: Dict[str, Any] = {"account_id": account_key}
            if metadata_filter:
                filter_dict.update(metadata_filter)
            vector_store.delete(filter=filter_dict)
            logger.info(
                "Deleted vectors for account %s with filter %s",
                account_key,
                metadata_filter,
            )
        except Exception as e:
            logger.error(
                f"Error deleting filtered vectors for account {account_id}: {e}"
            )
            raise
