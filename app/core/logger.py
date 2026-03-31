import logging

logger = logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(name)s | %(message)s"
    )
)  

def get_logger(name:str):
    return logging.getLogger(name.strip()) 