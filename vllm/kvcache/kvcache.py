import vineyard
from abc import ABC, abstractmethod
import hashlib
from typing import Optional, List, Tuple
from vllm.logger import init_logger
from collections import OrderedDict

import torch

logger = init_logger(__name__)

# The `KVCache` class is an abstract base class with methods for getting, putting, deleting, and
# prefetching data by prefix.
class KVCache(ABC):

    def __init__(self,
                 socket: str,
                 tensor_nbytes: int = 1024,
                 cache_capacity: int = 1024,
                 layer: int = 1,
                 rank: Optional[int] = None,
                 model: str = ''):
        self._socket = socket
        self._tensor_nbytes = tensor_nbytes
        self._cache_capacity = cache_capacity
        self._used_capacity = 0
        self._layer = layer
        self._rank = rank
        self._model = model

    def get_by_prefix(self,
                      prefix: List[int],
                      tokens: List[int],
                      kv_cache_list: List[List[Tuple[torch.Tensor,
                                                     torch.Tensor]]],
                      version: int = -1):
        """
        @param prefix The `prefix` parameter is a list of token as integers.
        @param tokens The `tokens` parameter is a list of integers.
        @param kv_cache_list The `kv_cache_list` parameter is a list of lists where each inner list
        contains tuples of two `torch.Tensor` objects. The inner lists represent key-value pairs with size of layers, and each tuple contains a key tensor and a corresponding value tensor. The outer lists reperesents the all the key-value lists of the same token prefix list, same size to the `token` parameter.
        @param version The `version` parameter is used to specify a particular version of the value. If no specific version is provided, the default version (-1) will be used. UNSUPPORTED for now.
        """
        raise NotImplementedError("get_by_prefix call is not implemented.")

    def put_by_prefix(self,
                      prefix: List[int],
                      tokens: List[int],
                      kv_cache_list: List[List[Tuple[torch.Tensor,
                                                     torch.Tensor]]],
                      version: int = -1):
        """  
        @param prefix The `prefix` parameter is a list of token as integers.
        @param tokens The `tokens` parameter is a list of integers.
        The inner lists represent key-value pairs with size of layers, and each tuple contains a key tensor and a corresponding value tensor. The outer lists reperesents the all the key-value lists of the same token prefix list, same size to the `token` parameter.
        @param version The `version` parameter is used to specify a particular version of the value. If no specific version is provided, the default version (-1) will be used. UNSUPPORTED for now.
        """
        raise NotImplementedError("put_by_prefix call is not implemented.")

    def delete_by_prefix(self,
                         prefix: List[int],
                         tokens: List[int],
                         kv_cache_list: List[List[Tuple[torch.Tensor,
                                                        torch.Tensor]]],
                         version: int = -1):
        raise NotImplementedError("delete_by_prefix call is not implemented.")

    def prefetch_by_prefix(self,
                           prefix: List[int],
                           tokens: List[int],
                           kv_cache_list: List[List[Tuple[torch.Tensor,
                                                          torch.Tensor]]],
                           version: int = -1):
        raise NotImplementedError(
            "prefetch_by_prefix call is not implemented.")


def ListHash(data: List[int]) -> str:
    return hashlib.md5(bytes(data)).hexdigest()


# The `VineyardBaseKVCache` class implements a key-value cache using Vineyard as the storage backend.
class VineyardBaseKVCache(KVCache):

    def __init__(self,
                 socket: str,
                 tensor_nbytes: int = 1024,
                 cache_capacity: int = 1024,
                 layer: int = 1,
                 rank: Optional[int] = None,
                 model: str = ''):
        super().__init__(socket, tensor_nbytes, cache_capacity, layer, rank,
                         model)
        self._vineyard_client = vineyard.connect(socket)
        self.consistent_hash = ListHash
        # a simple LRU implementation, need to store key, object ID and size of key-value pair.
        self._index = OrderedDict()

    def get_by_prefix(self,
                      prefix: List[int],
                      tokens: List[int],
                      kv_cache_list: List[List[Tuple[torch.Tensor,
                                                     torch.Tensor]]],
                      version: int = 1):
        for i in range(len(tokens)):
            tokenList = prefix + [tokens[i]]
            key = self.consistent_hash(tokenList)
            try:
                # will be returned as a tuple, need to convert to list
                value = self._vineyard_client.get(name=key)
                kv_cache_list[i] = [*value]
                self._index.move_to_end(key)
            except:
                kv_cache_list[i] = None
                logger.info("entry not found, key: %s, tokenList: [%s].", key,
                            ' '.join(map(str, tokenList)))

    def put_by_prefix(self,
                      prefix: List[int],
                      tokens: List[int],
                      kv_cache_list: List[List[Tuple[torch.Tensor,
                                                     torch.Tensor]]],
                      version: int = -1):
        tokenList = prefix.copy()
        for i in range(len(tokens)):
            tokenList.append(tokens[i])
            key = self.consistent_hash(tokenList)

            kv_size = 0
            for j in range(self._layer):
                kv_size += kv_cache_list[i][j][0].size(
                    dim=0) + kv_cache_list[i][j][1].size(dim=0)
            kv_size *= kv_cache_list[i][0][0].element_size()
            if (self._used_capacity + kv_size > self._cache_capacity):
                _, (evict_object_id,
                    removed_kv_size) = self._index.popitem(last=False)
                self._vineyard_client.delete(evict_object_id)
                self._used_capacity -= removed_kv_size
            insert_object_id = self._vineyard_client.put(
                name=key, value=kv_cache_list[i], persist=True)
            self._index[key] = (insert_object_id, kv_size)
            self._used_capacity += kv_size

    def remove_all(self):
        """
        Clears all elements from the vineyard server.
        """
        self._vineyard_client.clear()

    def __del__(self):
        self._vineyard_client.close()
