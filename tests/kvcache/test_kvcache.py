from vllm.kvcache.kvcache import VineyardBaseKVCache
import torch


def test_kvcache():
    layers = 2
    kvcache = VineyardBaseKVCache(socket='/tmp/vineyard.sock', layer=layers)
    prefix = [0, 1, 2, 3, 4]
    tokens = [5, 6, 7]
    # 2 layers
    kv_cache_list = [[(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))]]
    kvcache.put_by_prefix(prefix, tokens, kv_cache_list)
    returned_kv_cache_list = [None] * len(tokens)
    kvcache.get_by_prefix(prefix, tokens, returned_kv_cache_list)
    for i in range(len(tokens)):
        returned_kv_cache_list = [None]
        kvcache.get_by_prefix(prefix, [tokens[i]], returned_kv_cache_list)
        prefix.append(tokens[i])
        for j in range(layers):
            key1, value1 = returned_kv_cache_list[0][j]
            key2, value2 = kv_cache_list[i][j]
            assert torch.equal(key1, key2)
            assert torch.equal(value1, value2)


def test_kvcache_eviction():
    layers = 2
    kvcache = VineyardBaseKVCache(socket='/tmp/vineyard.sock',
                                  layer=layers,
                                  cache_capacity=200)
    prefix = [0, 1, 2, 3, 4]
    tokens = [5, 6, 7]
    # 2 layer
    kv_cache_list = [[(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))]]
    kvcache.put_by_prefix(prefix, tokens, kv_cache_list)

    # capacity should be 192, access sequence
    get_prefix = [0, 1, 2, 3, 4, 5]
    get_tokens = [6]
    returned_kv_cache_list = [None] * len(get_tokens)
    kvcache.get_by_prefix(get_prefix, get_tokens, returned_kv_cache_list)

    # insert another KV pair, which should trigger eviction of the first sequence
    prefix = [0, 1, 5, 6, 7]
    tokens = [10]
    kv_cache_list = [[(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))]]
    kvcache.put_by_prefix(prefix, tokens, kv_cache_list)

    # first sequence not found
    get_prefix = [0, 1, 2, 3, 4]
    get_tokens = [5]
    returned_kv_cache_list = [None] * len(get_tokens)
    kvcache.get_by_prefix(get_prefix, get_tokens, returned_kv_cache_list)
    assert (returned_kv_cache_list[0] == None)

    # the last sequence should be found
    get_prefix = [0, 1, 5, 6, 7]
    get_tokens = [10]
    returned_kv_cache_list = [None] * len(get_tokens)
    kvcache.get_by_prefix(get_prefix, get_tokens, returned_kv_cache_list)
    for j in range(layers):
        key1, value1 = returned_kv_cache_list[0][j]
        key2, value2 = kv_cache_list[0][j]
        assert torch.equal(key1, key2)
        assert torch.equal(value1, value2)


if __name__ == "__main__":
    test_kvcache()
    test_kvcache_eviction()
from vllm.kvcache.kvcache import VineyardBaseKVCache
import torch


def test_kvcache():
    layers = 2
    kvcache = VineyardBaseKVCache(socket='/tmp/vineyard.sock', layer=layers)
    prefix = [0, 1, 2, 3, 4]
    tokens = [5, 6, 7]
    # 2 layers
    kv_cache_list = [[(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))]]
    kvcache.put_by_prefix(prefix, tokens, kv_cache_list)
    returned_kv_cache_list = [None] * len(tokens)
    kvcache.get_by_prefix(prefix, tokens, returned_kv_cache_list)
    for i in range(len(tokens)):
        returned_kv_cache_list = [None]
        kvcache.get_by_prefix(prefix, [tokens[i]], returned_kv_cache_list)
        prefix.append(tokens[i])
        for j in range(layers):
            key1, value1 = returned_kv_cache_list[0][j]
            key2, value2 = kv_cache_list[i][j]
            assert torch.equal(key1, key2)
            assert torch.equal(value1, value2)


def test_kvcache_eviction():
    layers = 2
    kvcache = VineyardBaseKVCache(socket='/tmp/vineyard.sock',
                                  layer=layers,
                                  cache_capacity=200)
    prefix = [0, 1, 2, 3, 4]
    tokens = [5, 6, 7]
    # 2 layer
    kv_cache_list = [[(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))],
                     [(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))]]
    kvcache.put_by_prefix(prefix, tokens, kv_cache_list)

    # capacity should be 192, access sequence
    get_prefix = [0, 1, 2, 3, 4, 5]
    get_tokens = [6]
    returned_kv_cache_list = [None] * len(get_tokens)
    kvcache.get_by_prefix(get_prefix, get_tokens, returned_kv_cache_list)

    # insert another KV pair, which should trigger eviction of the first sequence
    prefix = [0, 1, 5, 6, 7]
    tokens = [10]
    kv_cache_list = [[(torch.rand(4), torch.rand(4)),
                      (torch.rand(4), torch.rand(4))]]
    kvcache.put_by_prefix(prefix, tokens, kv_cache_list)

    # first sequence not found
    get_prefix = [0, 1, 2, 3, 4]
    get_tokens = [5]
    returned_kv_cache_list = [None] * len(get_tokens)
    kvcache.get_by_prefix(get_prefix, get_tokens, returned_kv_cache_list)
    assert (returned_kv_cache_list[0] == None)

    # the last sequence should be found
    get_prefix = [0, 1, 5, 6, 7]
    get_tokens = [10]
    returned_kv_cache_list = [None] * len(get_tokens)
    kvcache.get_by_prefix(get_prefix, get_tokens, returned_kv_cache_list)
    for j in range(layers):
        key1, value1 = returned_kv_cache_list[0][j]
        key2, value2 = kv_cache_list[0][j]
        assert torch.equal(key1, key2)
        assert torch.equal(value1, value2)


if __name__ == "__main__":
    test_kvcache()
    test_kvcache_eviction()
