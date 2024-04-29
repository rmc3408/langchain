from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever

class RedundantRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma
    
    def get_relevant_documents(self, query):
        eb = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(embedding=eb, lambda_mult=0.8)
      
    def aget_relevant_documents(self, query):
        return []
