from typing import TypedDict

class RAGstate(TypedDict):
    question : str
    context: str
    sources: str
    answer : str
    Faithfullness : float
    Answer_releavency : float
    retry_count : int
    session_id: str
    time_taken: str
    
    


