from langgraph.graph import StateGraph, END
from app.graph.nodes import generate,query_rewritter,retriever
from app.graph.state import RAGstate
def build_graph():
    graph = StateGraph(RAGstate)
    
    graph.add_node("retriever",retriever.retrieval)
    graph.add_node("generator",generate.generator)
    graph.add_node("re_writter",query_rewritter.query_rewritter)
    
    
    graph.set_entry_point("retriever")
    graph.add_edge("retriever","generator")
    graph.add_edge("generator",END)
    return graph.compile()

graph =build_graph()
