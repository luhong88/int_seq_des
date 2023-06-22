import logging

logger= logging.getLogger(__name__)
logger.propagate= False
logger.setLevel(logging.DEBUG)
c_handler= logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)
sep= '-'*50


def get_array_chunk(arr, rank, size):
    chunk_size= len(arr)/size
    if not chunk_size.is_integer():
        raise ValueError(f'It is not possible to evenly divide an array of length {len(arr)} into {size + 1} processes.')
    else:
        chunk_size= int(chunk_size)
        
    if rank < size - 1:
        return arr[rank*chunk_size:(rank + 1)*chunk_size]
    else:
        return arr[rank*chunk_size:]